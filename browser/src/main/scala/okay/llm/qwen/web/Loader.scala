package okay.llm.qwen.web

import okay.llm.compute.{Compute, OnCpu, OnWebGl2, OnWebGpu}
import okay.llm.qwen.{Qwen3Compute, Qwen3Config, Rows, Safetensors, Tensors, Tokenizer, Vocab}
import okay.effects.json.Json

import scala.concurrent.{ExecutionContext, Future}
import scala.scalajs.js
import scala.scalajs.js.typedarray.*

/**
 * A MODEL INTO A PAGE (what okay-desk's page does, for any page): its files fetched once and kept on the device (the
 * Cache API), the device chosen — WebGPU, else WebGL2, else the CPU — and the forward made, reading no deeper than
 * the page needs.
 *
 * A model's directory: config.json, tokenizer.json, and either model.safetensors or manifest.json — `files` (the
 * weights in parts, in the order of their layers), and the embedding in blocks by script (`embedding`: a block → its
 * file, `index`: a byte a token, its block). `upTo`: the deepest layer read — the parts after it are not fetched (a
 * phone loads and computes fewer layers). `phone`: WebGL2 keeps a small pool of textures and the head stays on the
 * CPU (a phone's browser takes its GPU memory back at a few hundred MB).
 */
object Loader:
  /** a model loaded: the forward, its tokenizer, the device's name and why a better one was not had, and `ensure` —
   * the embedding's blocks of a sequence's tokens loaded before its forward */
  final case class Loaded(model: Qwen3Compute, tokenizer: Tokenizer, config: Qwen3Config, api: String,
      why: Option[String], ensure: Array[Int] => Future[Unit])

  /** the file at `url`, from the device's cache when it was fetched before; a quota refused is no failure */
  def fetch(url: String, cacheName: String)(using ExecutionContext): Future[ArrayBuffer] =
    val caches = js.Dynamic.global.caches
    val opened: Future[Option[js.Dynamic]] =
      if js.isUndefined(caches) then Future.successful(None)
      else caches.open(cacheName).asInstanceOf[js.Promise[js.Dynamic]].toFuture.map(Some(_)).recover { case _ => None }
    opened.flatMap { c =>
      val hit = c.fold(Future.successful(null: js.Dynamic))(cc =>
        cc.`match`(url).asInstanceOf[js.Promise[js.Dynamic]].toFuture.recover { case _ => null })
      hit.flatMap { r =>
        if r != null && !js.isUndefined(r) then r.arrayBuffer().asInstanceOf[js.Promise[ArrayBuffer]].toFuture
        else http(url).flatMap { r =>
          if !r.ok.asInstanceOf[Boolean] then Future.failed(IllegalStateException(s"$url: HTTP ${r.status}"))
          else
            val kept = c.fold(Future.unit)(cc =>
              cc.put(url, r.applyDynamic("clone")()).asInstanceOf[js.Promise[Unit]].toFuture.recover { case _ => () })
            kept.flatMap(_ => r.arrayBuffer().asInstanceOf[js.Promise[ArrayBuffer]].toFuture)
        }
      }
    }

  private def http(url: String): Future[js.Dynamic] =
    js.Dynamic.global.fetch(url).asInstanceOf[js.Promise[js.Dynamic]].toFuture

  def text(ab: ArrayBuffer): String =
    js.Dynamic.newInstance(js.Dynamic.global.TextDecoder)().decode(ab).asInstanceOf[String]

  /** the device: WebGPU (unless `webGpu` is false: a page trying its WebGL2), else WebGL2 (small on a phone), else
   * the CPU — and why the better ones were not had */
  def device(phone: Boolean, webGpu: Boolean = true)(using ExecutionContext): Future[(Compute, Option[String])] =
    (if webGpu then OnWebGpu.create() else Future.successful(Left("WebGPU not tried"))).map { gpu =>
      val chosen = gpu.left.flatMap(why =>
        OnWebGl2.create(if phone then 24L * 1048576 else 96L * 1048576, headsAtOnce = if phone then 2 else 16)
          .left.map(more => s"$why; $more"))
      chosen.fold(why => (OnCpu(): Compute, Some(why)), c => (c, gpu.left.toOption))
    }

  /** the deepest layer a part of the weights holds (−1 for none) */
  def deepest(st: Safetensors): Int =
    st.entries.keys.flatMap(n => "^model\\.layers\\.(\\d+)\\.".r.findFirstMatchIn(n).map(_.group(1).toInt))
      .maxOption.getOrElse(-1)

  /** the model of `dir` on `compute`, read no deeper than `upTo`; `scripts`: the embedding's blocks loaded at once
   * (others when a token of theirs appears); `stage`: what is being done, said as it goes */
  def load(dir: String, compute: Compute, why: Option[String], upTo: Int = Int.MaxValue, phone: Boolean = false,
      cacheName: String = "okay-llm-model", scripts: Vector[String] = Vector("common", "latin-en"),
      stage: String => Unit = _ => ())(using ExecutionContext): Future[Loaded] =
    def get(name: String) = fetch(s"$dir/$name", cacheName)
    for
      config <- get("config.json").map(text).map(Qwen3Config.parse)
      tok <- get("tokenizer.json").map(t => Tokenizer.fromJson(Json.fromText(text(t))))
      weighed <- weights(dir, config, upTo, cacheName, scripts, stage)
    yield
      val (st, blocks) = weighed
      stage(s"the weights to ${compute.api}…")
      val model = Qwen3Compute(compute, st, config, headOnCpu = phone && compute.api != "WebGPU",
        embedGiven = blocks.map(_._1))
      val ensure: Array[Int] => Future[Unit] = blocks match
        case Some((rows, index, more)) => ids =>
          val missing = ids.filter(id => id < index.length && !rows.has(id)).map(id => Vocab.blocks(index(id).toInt))
            .distinct
          if missing.isEmpty then Future.unit
          else missing.foldLeft(Future.unit)((acc, b) => acc.flatMap(_ => more(b))).map(_ => model.rehead())
        case None => _ => Future.unit
      Loaded(model, tok, config, compute.api, why, ensure)

  /** the weights' parts up to the one holding the layer after `upTo` (they come in the order of their layers), and
   * the embedding's blocks: the rows, the index, and how to load one more */
  private def weights(dir: String, config: Qwen3Config, upTo: Int, cacheName: String, scripts: Vector[String],
      stage: String => Unit)(using ExecutionContext)
      : Future[(Tensors, Option[(Rows.Blocks, Array[Byte], String => Future[Unit])])] =
    def get(name: String) = fetch(s"$dir/$name", cacheName)
    http(s"$dir/manifest.json").flatMap { r =>
      if !r.ok.asInstanceOf[Boolean] then
        get("model.safetensors").map(b => (new Safetensors(TypedArrayBuffer.wrap(b)), None))
      else r.text().asInstanceOf[js.Promise[String]].toFuture.flatMap { t =>
        val m = Json.fromText(t)
        val files = m.field("files").fold(Vector.empty[String])(_.list.map(_.string))
        val blockFiles = m.field("embedding")
          .fold(Map.empty[String, String])(_.record.map((k, v) => k -> v.string).toMap)
        def parts(rest: List[(String, Int)], got: Vector[Tensors]): Future[Vector[Tensors]] = rest match
          case Nil => Future.successful(got)
          case (f, i) :: more =>
            stage(s"the weights: part ${i + 1} of ${files.size}…")
            get(f).flatMap { b =>
              val st = new Safetensors(TypedArrayBuffer.wrap(b))
              if deepest(st) > upTo then Future.successful(got :+ st) else parts(more, got :+ st)
            }
        val embedding =
          if blockFiles.isEmpty then Future.successful(None)
          else
            val indexFile = m.field("index").fold("embed-index.bin")(_.string)
            get(indexFile).flatMap { ib =>
              val rows = Rows.Blocks(config.vocab, config.hidden)
              def block(name: String): Future[Unit] = blockFiles.get(name).fold(Future.unit) { file =>
                stage(s"the embedding: the $name block…")
                get(file).map { b =>
                  val (ids, q) = Vocab.readBlock(new Safetensors(TypedArrayBuffer.wrap(b))); rows.add(ids, q)
                }
              }
              scripts.foldLeft(Future.unit)((acc, b) => acc.flatMap(_ => block(b)))
                .map(_ => Some((rows, new Int8Array(ib).toArray, block)))
            }
        for ps <- parts(files.toList.zipWithIndex, Vector.empty); e <- embedding yield (Tensors.Many(ps), e)
      }
    }
