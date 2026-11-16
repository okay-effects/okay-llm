package okay.llm.read

import okay.llm.qwen.{Qwen3Compute, Tokenizer}

import scala.concurrent.{ExecutionContext, Future}

/**
 * THE BROWSER'S READING (okay-desk specs/data.md §15, step 4): `Read` answered by okay-llm's Qwen3 forward on a
 * `Compute` device — WebGPU, WebGL2 or the CPU — each a future (`Read.future`). The chat prompt is Qwen3's template
 * with thinking off, as mlx-scala renders it on the JVM: the same tokens, so the skill trained there reads here.
 *
 * ONE PREFIX CACHED (§17): the readers ask ten to fifteen times about the same prompt with a few tokens appended
 * (`{"from": "orders", "where": [{"column": "`); the first ask of a sequence runs it whole and keeps its keys,
 * values and states on the device, every later ask that begins with it runs only the tokens after — a question
 * costs one forward of the prompt, not fifteen. A sequence that does not begin with the cached one replaces it.
 */
/** `ensure`: the embedding's rows of these tokens loaded before a forward (blocks by language fetched as needed) */
final class OnQwen(model: Qwen3Compute, tok: Tokenizer, layer: Int,
    ensure: Array[Int] => Future[Unit] = _ => Future.unit)(using ExecutionContext):
  def rehead(): Unit = model.rehead()
  def clearCache(): Unit =
    cached.foreach(_._2.free()); cached = None
  def close(): Unit =
    clearCache()
    model.c.close()
  private var cached: Option[(Array[Int], model.Cache)] = None
  /** the forward: after the cached prefix when `ids` begins with it, else whole and kept */
  private def run(ids: Array[Int], at: Seq[Int], withLogits: Boolean): Future[(Vector[Array[Float]], Array[Float])] =
    ensure(ids).flatMap(_ => forward(ids, at, withLogits))
  private def forward(ids: Array[Int], at: Seq[Int], withLogits: Boolean)
      : Future[(Vector[Array[Float]], Array[Float])] =
    cached match
      // (the cached sequence itself asked again: run whole — the cache holds no final state for its last row)
      case Some((prefix, cache)) if ids.length > prefix.length && ids.startsWith(prefix) =>
        val attempted = try
          model.run(ids, layer, at, withLogits, Some(cache), keep = withLogits, consumeCache = withLogits)
        catch case error: Exception => Future.failed(error)
        attempted.map { (s, l, next) =>
          if withLogits then
            cached = next.map(ids -> _)
            cache.free()
          (s, l)
        }.recoverWith { case error => clearCache(); Future.failed(error) }
      case _ =>
        cached.foreach(_._2.free()); cached = None
        model.run(ids, layer, at, withLogits, None, keep = true)
          .map { (s, l, c) => cached = c.map(ids -> _); (s, l) }
  def answer[X](op: Read[X]): Future[X] = op match
    case Read.States(ids, at) => run(ids, at, withLogits = false).map(_._1)
    case Read.Logits(ids) => run(ids, Nil, withLogits = true).map(_._2)
    case Read.Encode(text) => Future.successful(tok.encode(text))
    case Read.Decode(ids) => Future.successful(tok.decode(ids))
    case Read.Prompt(system, user) => Future.successful(tok.encode(OnQwen.template(system, user)))
    case Read.Note(_) => Future.unit
    // a line's options weighed in one packed forward after the context's cache (okay-llm's `Qwen3Compute.score`);
    // the log-softmax within the options' tokens (the head a few hundred rows: a phone's; Everyday 62 of 108 right as
    // over the whole vocabulary, 0.91 s a question on the JVM)
    case Read.Score(context, options) =>
      val head = context.init
      // (a def: run only after the blocks are loaded)
      def ready: Future[model.Cache] = cached match
        case Some((prefix, cache)) if prefix.sameElements(head) => Future.successful(cache)
        case Some((prefix, cache)) if head.startsWith(prefix) =>
          model.run(head, layer, Nil, withLogits = false, Some(cache), keep = true).map { (_, _, next) =>
            cache.free(); cached = next.map(head -> _); next.get }
        case _ =>
          clearCache()
          model.run(head, layer, Nil, withLogits = false, None, keep = true).map { (_, _, next) =>
            cached = next.map(head -> _); next.get }
      // (the options' tokens' embedding blocks loaded first: the head knows only the loaded rows)
      ensure(context ++ options.flatten).flatMap(_ => ready)
        .flatMap(cache => model.score(cache, context.last, options, within = true))
  def handler: [X] => Read[X] => Future[X] = [X] => (op: Read[X]) => answer[X](op)

  /**
   * THE STATES OF ONE READING, KEEPING NOTHING (okay-desk specs/formalization.md §18.3): a statement reader asks once
   * about a prompt no later ask begins with — so no prefix is kept, and each layer's keys and values are let go as
   * soon as the next layer is computed. On a phone that is the difference between all the layers' keys and values of a long
   * task at once (some 100 MB) and one layer's. The cache of other readers is dropped first.
   */
  def statesOnce(ids: Array[Int], at: Seq[Int]): Future[Vector[Array[Float]]] =
    clearCache()
    ensure(ids).flatMap(_ => model.run(ids, layer, at, withLogits = false, None, keep = false)).map(_._1)
  /** `c.api` of the device it runs on */
  def api: String = model.c.api
  def layers: Int = model.config.layers

object OnQwen:
  /** Qwen3's chat template, thinking off (an empty think block, as the template writes it) */
  def template(system: String, user: String): String =
    s"<|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n" +
      "<|im_start|>assistant\n<think>\n\n</think>\n\n"
