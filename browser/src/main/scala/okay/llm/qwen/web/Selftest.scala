package okay.llm.qwen.web

import okay.effects.json.Json
import okay.llm.compute.{Compute, OnCpu, OnWebGl2, OnWebGpu}
import okay.llm.qwen.{Qwen3Compute, Qwen3Config, Safetensors, Tokenizer}

import scala.concurrent.Future
import scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
import scala.scalajs.js
import scala.scalajs.js.Dynamic.global as g
import scala.scalajs.js.typedarray.*

/**
 * THE PAGE'S SELF-TEST (browser/index.html, `?model=<dir url>`): the model's files fetched (config.json,
 * tokenizer.json, model.safetensors), a prompt tokenized in the page, the forward on WebGPU and on WebGL2 against
 * the CPU handler in the same page — the states after the last layer at the first and the last position, the
 * logits, the next token — the largest difference and the time of each, written into the page (#selftest).
 */
object Selftest:
  def main(args: Array[String]): Unit =
    val out = g.document.createElement("pre"); out.id = "selftest"
    g.document.body.appendChild(out)
    def say(s: String): Unit = { out.textContent = out.textContent.toString + s + "\n"; println(s) }
    val now = () => g.performance.now().asInstanceOf[Double]
    val params = js.Dynamic.newInstance(g.URLSearchParams)(g.location.search)
    val dir = Option(params.get("model").asInstanceOf[String]).getOrElse("model")
    def fetch(name: String): Future[ArrayBuffer] =
      g.fetch(s"$dir/$name").asInstanceOf[js.Promise[js.Dynamic]].toFuture.flatMap { r =>
        if !r.ok.asInstanceOf[Boolean] then Future.failed(IllegalStateException(s"$name: HTTP ${r.status}"))
        else r.arrayBuffer().asInstanceOf[js.Promise[ArrayBuffer]].toFuture
      }
    def text(ab: ArrayBuffer): String = js.Dynamic.newInstance(g.TextDecoder)().decode(ab).asInstanceOf[String]
    (for
      configText <- fetch("config.json").map(text)
      tokText <- fetch("tokenizer.json").map(text)
      model <- fetch("model.safetensors")
      config = Qwen3Config.parse(configText)
      tok = Tokenizer.fromJson(Json.fromText(tokText))
      st = new Safetensors(TypedArrayBuffer.wrap(model))
      prompt = "<|im_start|>user\nHow many rows have country = France?<|im_end|>\n<|im_start|>assistant\n"
      ids = tok.encode(prompt)
      _ = say(s"model: ${config.layers} layers, ${model.byteLength / 1048576} MB; the prompt ${ids.length} tokens: " +
        ids.mkString(" "))
      at = Seq(0, ids.length - 1)
      run = (c: Compute) =>
        val t0 = now(); val m = Qwen3Compute(c, st, config); val t1 = now()
        m.forward(ids, config.layers, at, withLogits = true).map { case (states, logits) =>
          val t2 = now()
          val next = logits.indexOf(logits.max)
          say(f"${c.api}: weights up ${(t1 - t0) / 1000}%.1f s, forward ${(t2 - t1) / 1000}%.1f s; next token $next " +
            s"'${tok.decode(Seq(next))}'")
          (states, logits, next)
        }
      cpu <- run(OnCpu())
      check = (name: String, got: (Vector[Array[Float]], Array[Float], Int)) =>
        val sd = cpu._1.zip(got._1).map((a, b) => a.zip(b).map((x, y) => math.abs(x - y)).max)
        val ld = cpu._2.zip(got._2).map((x, y) => math.abs(x - y)).max
        val scale = cpu._1.map(_.map(math.abs).max).max
        say(f"$name vs CPU: states' max abs diff ${sd.mkString(" ")} (the states' max $scale%.0f); logits' $ld%.3f; " +
          s"next token ${if got._3 == cpu._3 then "the same" else "DIFFERENT"}")
        sd.forall(_ < 0.02f * scale + 0.05f) && ld < 0.5f && got._3 == cpu._3
      gpu <- OnWebGpu.create()
      okGpu <- gpu.fold(why => { say(s"WebGPU: $why"); Future.successful(true) },
        d => { say(s"WebGPU: ${d.adapter}"); run(d).map(check("WebGPU", _)) })
      gl = OnWebGl2.create()
      okGl <- gl.fold(why => { say(s"WebGL2: $why"); Future.successful(true) },
        d => { say(s"WebGL2: ${d.adapter}"); run(d).map(check("WebGL2", _)) })
    yield say(if okGpu && okGl then "selftest: done" else "selftest: failed: a device differs from the CPU"))
      .recover { case e => say(s"selftest: failed: $e"); e.printStackTrace() }
