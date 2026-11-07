package okay.llm.qwen

import okay.llm.compute.OnCpu

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Await
import scala.concurrent.duration.*

/** THE HEAD THE FILE HAS: a file with its own `lm_head` (here a copy of the embedding) gives the logits of the tied
 * one — on the device and on the CPU head alike. On a model from `$QWEN_DIR`; skipped without one. */
class UntiedHeadSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("an untied head that copies the embedding == the tied head") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val bytes = java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors"))
    val st = Safetensors(bytes)
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
    val names = st.entries.keys.toVector.sorted
    val copied = Safetensors(Safetensors.write(names.map(n => (n, st.entries(n).dtype, st.entries(n).shape, st.raw(n))) ++
      Vector("weight", "scales", "biases").map(p => (s"lm_head.$p", st.entries(s"model.embed_tokens.$p").dtype, st.entries(s"model.embed_tokens.$p").shape, st.raw(s"model.embed_tokens.$p")))))
    assert(copied.has("lm_head.weight"))
    for onCpu <- Vector(false, true) do
      val tied = Await.result(Qwen3Compute(OnCpu(), st, config, headOnCpu = onCpu).forward(ids, 0, Nil, withLogits = true), 10.minutes)._2
      val own = Await.result(Qwen3Compute(OnCpu(), copied, config, headOnCpu = onCpu).forward(ids, 0, Nil, withLogits = true), 10.minutes)._2
      val diff = tied.zip(own).map((a, b) => math.abs(a - b)).max
      println(s"untied vs tied (head on cpu $onCpu): max abs diff $diff")
      assert(diff < 1e-3f, diff)
  }

/** THE CACHE'S LAW: a prefix run and kept, then the rest after it — the same states and logits as the whole run at
 * once, exactly, on the CPU handler */
class CacheSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("a cached prefix then the rest == the whole at once") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
    val layer = math.min(config.layers, 19)
    val m = Qwen3Compute(OnCpu(), st, config)
    val at = Seq(0, 5, ids.length - 1)
    val (whole, logits) = Await.result(m.forward(ids, layer, at, withLogits = true), 10.minutes)
    val (_, _, cache) = Await.result(m.run(ids.take(12), layer, Nil, withLogits = false, None, keep = true), 10.minutes)
    val (got, gotLogits, _) = Await.result(m.run(ids, layer, at, withLogits = true, cache, keep = false), 10.minutes)
    assertEquals(got.size, whole.size)
    val diff = whole.zip(got).map((a, b) => a.zip(b).map((x, y) => math.abs(x - y)).max)
    val ldiff = logits.zip(gotLogits).map((x, y) => math.abs(x - y)).max
    println(s"cache: states' max abs diff ${diff.mkString(" ")}; logits' $ldiff")
    assert(diff.forall(_ < 1e-3f), diff); assert(ldiff < 1e-3f, ldiff)
    // nothing new after the cached whole: the states from the cache alone
    val (_, _, all) = Await.result(m.run(ids, layer, Nil, withLogits = false, None, keep = true), 10.minutes)
    val (same, _, _) = Await.result(m.run(ids, layer, at, withLogits = false, all, keep = false), 10.minutes)
    assert(whole.zip(same).forall((a, b) => a.sameElements(b)))
  }

/** the head on the CPU: the same logits as on the device */
class HeadSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("logits by the CPU head == logits by the device's") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
    val a = Await.result(Qwen3Compute(OnCpu(), st, config).forward(ids, 0, Nil, withLogits = true), 10.minutes)._2
    val b = Await.result(Qwen3Compute(OnCpu(), st, config, headOnCpu = true).forward(ids, 0, Nil, withLogits = true), 10.minutes)._2
    assert(a.zip(b).map((x, y) => math.abs(x - y)).max < 1e-3f)
  }

/** the file split into files of a few layers each, read as one: every tensor the same */
class SplitSuite extends munit.FunSuite:
  test("split and read as Many == the whole") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val files = Safetensors.split(st, 60 * 1048576)
    println(s"split: ${files.size} files of ${files.map(_.length / 1048576).mkString(", ")} MB")
    assert(files.size >= 4 && files.forall(_.length <= 90 * 1048576))
    val many = Tensors.Many(files.map(Safetensors(_)))
    for n <- st.entries.keys do
      assert(many.has(n), n)
      assertEquals(many.shape(n), st.shape(n), n)
      if st.entries(n).dtype == "U32" then assert(many.words(n).sameElements(st.words(n)), n)
      else assert(many.floats(n).sameElements(st.floats(n)), n)
  }

/** THE VOCABULARY'S BLOCKS: every block written and read, loaded together == the whole embedding (the states, the
 * logits, exactly); the blocks of English and Russian alone: the logits of their tokens the same, the rest −∞ */
class VocabSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("blocks together == the whole; a language's blocks alone") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val tok = Tokenizer.fromJson(okay.effects.json.Json.fromText(java.nio.file.Files.readString(dir.get.resolve("tokenizer.json"))))
    val words = java.nio.file.Path.of("/usr/share/dict/words")
    val uses = Vocab.uses(tok, if java.nio.file.Files.exists(words) then scala.io.Source.fromFile(words.toFile).getLines()
      else Iterator("How", "many", "orders", "are", "unpaid", "total", "amount", "per", "country", "user", "assistant"))
    val of = Vocab.of(tok, uses)
    val byBlock = (0 until tok.size).groupBy(of).view.mapValues(_.toVector).toMap
    println(s"blocks: " + Vocab.blocks.map(b => s"$b ${byBlock.getOrElse(b, Vector.empty).size}").mkString(", "))
    val files = byBlock.map((b, ids) => b -> Vocab.blockFile(st, ids))
    val all = Rows.Blocks(config.vocab, config.hidden)   // the embedding has rows past the tokenizer's size (padding)
    for b <- Vocab.blocks; f <- files.get(b) do { val (ids, q) = Vocab.readBlock(Safetensors(f)); all.add(ids, q) }
    val ids = tok.encode("<|im_start|>user\nHow many orders are unpaid? Сколько заказов?<|im_end|>\n<|im_start|>assistant\n")
    val whole = Qwen3Compute(OnCpu(), st, config)
    val blocks = Qwen3Compute(OnCpu(), st, config, embedGiven = Some(all))
    val (s1, l1) = Await.result(whole.forward(ids, config.layers, Seq(0, ids.length - 1), withLogits = true), 10.minutes)
    val (s2, l2) = Await.result(blocks.forward(ids, config.layers, Seq(0, ids.length - 1), withLogits = true), 10.minutes)
    assert(s1.zip(s2).forall((a, b) => a.sameElements(b)))
    assert(l1.take(tok.size).sameElements(l2.take(tok.size)))
    // English and Russian alone: common, latin-en, cyrillic
    val some = Rows.Blocks(config.vocab, config.hidden)
    for b <- Vector("common", "latin-en", "cyrillic"); f <- files.get(b) do { val (i, q) = Vocab.readBlock(Safetensors(f)); some.add(i, q) }
    val missing = ids.filterNot(some.has).map(id => s"$id '${tok.decode(Seq(id))}' ${of(id)}")
    assert(missing.isEmpty, "tokens of the prompt with no block loaded: " + missing.mkString(", "))
    val partial = Qwen3Compute(OnCpu(), st, config, headOnCpu = true, embedGiven = Some(some))
    val (s3, l3) = Await.result(partial.forward(ids, config.layers, Seq(0, ids.length - 1), withLogits = true), 10.minutes)
    assert(s1.zip(s3).forall((a, b) => a.sameElements(b)))
    val cjk = (0 until tok.size).find(id => of(id) == "cjk").get
    assert(l3(cjk).isNegInfinity && l3(ids.last) == l1(ids.last))
    println(f"loaded for English + Russian: ${some.packed._2.length} of ${tok.size} rows")
  }

/** A COMPONENT ABSENT == A COMPONENT SILENCED: a layer's attention (and another's MLP) taken out of the file gives
 * the same states as the same component left in with its output's scales and biases zeroed (what the map did) */
class SkipSuite extends munit.FunSuite:
  override val munitTimeout = 10.minutes
  test("the file without a layer's attention and another's MLP == the file with them zeroed") {
    val dir = sys.env.get("QWEN_DIR").map(java.nio.file.Path.of(_))
    assume(dir.isDefined, "QWEN_DIR not set")
    val st = Safetensors(java.nio.file.Files.readAllBytes(dir.get.resolve("model.safetensors")))
    val config = Qwen3Config.parse(java.nio.file.Files.readString(dir.get.resolve("config.json")))
    val attnOf = "model.layers.1.self_attn."; val mlpOf = "model.layers.9.mlp."
    val zeroedNames = Set(attnOf + "o_proj.scales", attnOf + "o_proj.biases", mlpOf + "down_proj.scales", mlpOf + "down_proj.biases")
    def entry(n: String, bytes: Array[Byte]) = { val e = st.entries(n); (n, e.dtype, e.shape, bytes) }
    val zeroed = Safetensors(Safetensors.write(st.entries.keys.toVector.sorted.map(n =>
      entry(n, if zeroedNames(n) then new Array[Byte](st.raw(n).length) else st.raw(n)))))
    val absent = Safetensors(Safetensors.write(st.entries.keys.toVector.sorted
      .filterNot(n => n.startsWith(attnOf) || n == "model.layers.1.input_layernorm.weight" || n.startsWith(mlpOf) || n == "model.layers.9.post_attention_layernorm.weight")
      .map(n => entry(n, st.raw(n)))))
    val ids = Array(151644, 872, 198, 4340, 1657, 6978, 614, 3146, 284, 9625, 30, 151645, 198, 151644, 77091, 198)
    val a = Await.result(Qwen3Compute(OnCpu(), zeroed, config).forward(ids, config.layers, Seq(0, 7, ids.length - 1), withLogits = true), 10.minutes)
    val b = Await.result(Qwen3Compute(OnCpu(), absent, config).forward(ids, config.layers, Seq(0, 7, ids.length - 1), withLogits = true), 10.minutes)
    assert(a._1.zip(b._1).forall((x, y) => x.sameElements(y)), "the states")
    assert(a._2.sameElements(b._2), "the logits")
    println("a component absent == silenced: the states and the logits the same, exactly")
  }
