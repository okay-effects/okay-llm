package okay.llm.compute

import scala.concurrent.{ExecutionContext, Future}
import scala.scalajs.js
import scala.scalajs.js.Dynamic.{global as g, literal}
import scala.scalajs.js.typedarray.*

/**
 * THE DEVICE ON WEBGPU: a tensor a storage buffer, a weight three (the words, the scales, the biases), an op a
 * compute dispatch (`Wgsl`) recorded into one command encoder, submitted when something is read back. A released
 * buffer is destroyed once the work submitted so far is done — so a tensor may be released as soon as its reader
 * is issued, as `Compute` promises.
 */
final class OnWebGpu private (device: js.Dynamic, val adapter: String)(using ExecutionContext) extends Compute:
  val api = "WebGPU"
  override def close(): Unit =
    device.destroy()
    bytes = 0L
  final class T(val rows: Int, val cols: Int, val buffer: js.Dynamic)
  final class W(val out: Int, val in: Int, val words: js.Dynamic, val scales: js.Dynamic, val biases: js.Dynamic)
  private val STORAGE = 0x80; private val COPY_DST = 0x08; private val COPY_SRC = 0x04; private val UNIFORM = 0x40
  private val MAP_READ = 0x01
  /** the bytes held in buffers now */
  var bytes = 0L

  private def buffer(size: Int, usage: Int): js.Dynamic =
    val n = math.max(16, (size + 3) / 4 * 4)
    bytes += n
    device.createBuffer(literal(size = n, usage = usage))
  private def upload(data: js.typedarray.TypedArray[?, ?], bytesLength: Int): js.Dynamic =
    val b = buffer(bytesLength, STORAGE | COPY_DST | COPY_SRC)
    device.queue.writeBuffer(b, 0, data)
    b
  def tensor(rows: Int, cols: Int, data: Array[Float]): T = T(rows, cols, upload(data.toTypedArray, data.length * 4))
  def quantized(q: QLinear): W = W(q.out, q.in, upload(q.words.toTypedArray, q.words.length * 4),
    upload(q.scales.toTypedArray, q.scales.length * 4), upload(q.biases.toTypedArray, q.biases.length * 4))

  private def pipeline(code: String): js.Dynamic =
    device.createComputePipeline(literal(layout = "auto",
      compute = literal(module = device.createShaderModule(literal(code = code)), entryPoint = "main")))
  private lazy val linearP = pipeline(Wgsl.linear); private lazy val normP = pipeline(Wgsl.rmsNorm)
  private lazy val ropeP = pipeline(Wgsl.headNormRope); private lazy val scoresP = pipeline(Wgsl.scores)
  private lazy val attendP = pipeline(Wgsl.attend); private lazy val addP = pipeline(Wgsl.add)
  private lazy val swigluP = pipeline(Wgsl.swiglu)

  private var enc: js.Dynamic = null
  private val pending = scala.collection.mutable.ArrayBuffer.empty[(js.Dynamic, Int)]
  private def encoder(): js.Dynamic = { if enc == null then enc = device.createCommandEncoder(); enc }
  private def destroyLater(b: js.Dynamic, size: Int): Unit = pending += ((b, size))
  /** the work so far submitted; what was released is destroyed when it is done */
  private def submit(): Unit = if enc != null then
    device.queue.submit(js.Array(enc.finish())); enc = null
    val free = pending.toVector; pending.clear()
    device.queue.onSubmittedWorkDone().asInstanceOf[js.Promise[Unit]].toFuture.foreach { _ =>
      free.foreach((b, size) => { b.destroy(); bytes -= size })
    }
  def release(x: T): Unit = destroyLater(x.buffer, math.max(16, x.rows * x.cols * 4))

  /** the parameters of a dispatch: four words and four floats, a uniform buffer destroyed after the submit */
  private def params(ints: Seq[Int], floats: Seq[Float]): js.Dynamic =
    val b = device.createBuffer(literal(size = 32, usage = UNIFORM | COPY_DST)); bytes += 32
    val u = new Uint32Array(8); val f = new Float32Array(u.buffer)
    ints.zipWithIndex.foreach((w, i) => u(i) = w); floats.zipWithIndex.foreach((w, i) => f(4 + i) = w)
    device.queue.writeBuffer(b, 0, u)
    destroyLater(b, 32)
    b
  private def dispatch(p: js.Dynamic, buffers: Seq[js.Dynamic], x: Int, y: Int = 1, z: Int = 1): Unit =
    val group = device.createBindGroup(literal(layout = p.getBindGroupLayout(0),
      entries = js.Array(buffers.zipWithIndex.map((b, i) => literal(binding = i, resource = literal(buffer = b)))*)))
    val pass = encoder().beginComputePass()
    pass.setPipeline(p); pass.setBindGroup(0, group); pass.dispatchWorkgroups(x, y, z); pass.end()
  private def elementwise(p: js.Dynamic, a: T, b: T): T =
    val out = T(a.rows, a.cols, buffer(a.rows * a.cols * 4, STORAGE | COPY_SRC | COPY_DST))
    val size = a.rows * a.cols; val wg = (size + 63) / 64
    dispatch(p, Seq(params(Seq(size), Nil), a.buffer, b.buffer, out.buffer), math.min(wg, 65535), (wg + 65534) / 65535)
    out

  def linear(x: T, w: W): T =
    val out = T(x.rows, w.out, buffer(x.rows * w.out * 4, STORAGE | COPY_SRC | COPY_DST))
    dispatch(linearP, Seq(params(Seq(x.rows, w.in, w.out), Nil), x.buffer, w.words, w.scales, w.biases, out.buffer),
      (w.out + 63) / 64, x.rows)
    out
  def rmsNorm(x: T, w: T, eps: Float): T =
    val out = T(x.rows, x.cols, buffer(x.rows * x.cols * 4, STORAGE | COPY_SRC | COPY_DST))
    dispatch(normP, Seq(params(Seq(x.rows, x.cols), Seq(eps)), x.buffer, w.buffer, out.buffer), (x.rows + 63) / 64)
    out
  def headNormRope(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float, pos0: Int): T =
    val out = T(x.rows, x.cols, buffer(x.rows * x.cols * 4, STORAGE | COPY_SRC | COPY_DST))
    dispatch(ropeP, Seq(params(Seq(x.rows, heads, headDim, pos0), Seq(eps, theta)), x.buffer, w.buffer, out.buffer),
      (heads + 63) / 64, x.rows)
    out
  def attention(q: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int, pos0: Int): T =
    val n = q.rows; val nk = k.rows
    require(nk == pos0 + n, s"k has $nk rows for $pos0 + $n")
    val scores = buffer(n * heads * nk * 4, STORAGE)
    val p = params(Seq(n, heads, kvHeads, headDim), Seq((1.0 / math.sqrt(headDim)).toFloat, pos0.toFloat))
    dispatch(scoresP, Seq(p, q.buffer, k.buffer, scores), (nk + 63) / 64, n, heads)
    val out = T(n, heads * headDim, buffer(n * heads * headDim * 4, STORAGE | COPY_SRC | COPY_DST))
    dispatch(attendP, Seq(p, scores, v.buffer, out.buffer), (headDim + 63) / 64, n, heads)
    destroyLater(scores, n * heads * nk * 4)
    out
  def concat(a: T, b: T): T =
    val out = T(a.rows + b.rows, a.cols, buffer((a.rows + b.rows) * a.cols * 4, STORAGE | COPY_SRC | COPY_DST))
    if a.rows > 0 then encoder().copyBufferToBuffer(a.buffer, 0, out.buffer, 0, a.rows * a.cols * 4)
    if b.rows > 0 then encoder().copyBufferToBuffer(b.buffer, 0, out.buffer, a.rows * a.cols * 4, b.rows * b.cols * 4)
    out
  def add(a: T, b: T): T = elementwise(addP, a, b)
  def swiglu(gate: T, up: T): T = elementwise(swigluP, gate, up)
  def slice(x: T, from: Int, until: Int): T =
    val out = T(until - from, x.cols, buffer((until - from) * x.cols * 4, STORAGE | COPY_SRC | COPY_DST))
    encoder().copyBufferToBuffer(x.buffer, from * x.cols * 4, out.buffer, 0, (until - from) * x.cols * 4)
    out
  def rows(x: T, at: Seq[Int]): Future[Vector[Array[Float]]] =
    val size = at.size * x.cols * 4
    val staging = buffer(size, MAP_READ | COPY_DST)
    at.zipWithIndex.foreach((r, k) => encoder().copyBufferToBuffer(x.buffer, r * x.cols * 4, staging, k * x.cols * 4, x.cols * 4))
    submit()
    staging.mapAsync(MAP_READ).asInstanceOf[js.Promise[Unit]].toFuture.map { _ =>
      val m = new Float32Array(staging.getMappedRange().asInstanceOf[ArrayBuffer])
      val all = m.toArray
      staging.unmap(); staging.destroy(); bytes -= math.max(16, size)
      at.indices.toVector.map(k => all.slice(k * x.cols, (k + 1) * x.cols))
    }

object OnWebGpu:
  /** the device of the page's GPU, if it has WebGPU */
  def create()(using ExecutionContext): Future[Either[String, OnWebGpu]] =
    val gpu = g.navigator.gpu
    if js.isUndefined(gpu) || gpu == null then
      Future.successful(Left("this browser has no WebGPU (navigator.gpu is missing" +
        (if g.isSecureContext.asInstanceOf[Boolean] then ")" else "; the page is not a secure context)")))
    else gpu.requestAdapter().asInstanceOf[js.Promise[js.Dynamic]].toFuture.flatMap { adapter =>
      if adapter == null then Future.successful(Left("WebGPU is here but gave no GPU adapter"))
      else
        val limits = adapter.limits; val info = adapter.info
        def field(k: String) = if js.isUndefined(info) || js.isUndefined(info.selectDynamic(k)) then "" else
          info.selectDynamic(k).toString
        val name = Seq(field("vendor"), field("architecture"), field("description")).filter(_.nonEmpty).mkString(" ")
        adapter.requestDevice(literal(requiredLimits = literal(
          maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize, maxBufferSize = limits.maxBufferSize,
          maxComputeWorkgroupsPerDimension = limits.maxComputeWorkgroupsPerDimension)))
          .asInstanceOf[js.Promise[js.Dynamic]].toFuture
          .map(d => Right(OnWebGpu(d, if name.isEmpty then "a GPU" else name)))
    }.recover { case e => Left(s"WebGPU failed: ${e.getMessage}") }

/** THE KERNELS, in WGSL: the parameters a uniform of four words and four floats; a thread an output element, or a
 * row where a row's sum is needed first */
object Wgsl:
  private val head = "struct P { a: vec4<u32>, b: vec4<f32> };\n@group(0) @binding(0) var<uniform> p: P;\n"
  val linear: String = head + """
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> words: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read> biases: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let o = id.x; let r = id.y; let inn = p.a.y; let out = p.a.z;
  if (o >= out) { return; }
  let groups = inn / 64u; let perRow = inn / 8u;
  var acc = 0.0;
  for (var gi = 0u; gi < groups; gi++) {
    let s = scales[o * groups + gi]; let b = biases[o * groups + gi];
    for (var k = 0u; k < 8u; k++) {
      let word = words[o * perRow + gi * 8u + k];
      let base = r * inn + gi * 64u + k * 8u;
      for (var j = 0u; j < 8u; j++) {
        acc += (s * f32((word >> (j * 4u)) & 15u) + b) * x[base + j];
      }
    }
  }
  y[r * out + o] = acc;
}"""
  val rmsNorm: String = head + """
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x; if (r >= p.a.x) { return; }
  let d = p.a.y; var ss = 0.0;
  for (var i = 0u; i < d; i++) { let v = x[r * d + i]; ss += v * v; }
  let s = 1.0 / sqrt(ss / f32(d) + p.b.x);
  for (var i = 0u; i < d; i++) { y[r * d + i] = x[r * d + i] * s * w[i]; }
}"""
  val headNormRope: String = head + """
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let hd = id.x; let r = id.y; let heads = p.a.y; let dim = p.a.z; let pos = f32(p.a.w + r);
  if (hd >= heads) { return; }
  let base = r * heads * dim + hd * dim; let half = dim / 2u;
  var ss = 0.0;
  for (var i = 0u; i < dim; i++) { let v = x[base + i]; ss += v * v; }
  let s = 1.0 / sqrt(ss / f32(dim) + p.b.x);
  for (var i = 0u; i < half; i++) {
    let x1 = x[base + i] * s * w[i]; let x2 = x[base + half + i] * s * w[half + i];
    let a = pos * pow(p.b.y, -2.0 * f32(i) / f32(dim));
    let c = cos(a); let sn = sin(a);
    y[base + i] = x1 * c - x2 * sn; y[base + half + i] = x2 * c + x1 * sn;
  }
}"""
  /** the scores s[t, h, j] = q[t, h] · k[j, kv(h)] scaled, -1e30 beyond the causal edge */
  val scores: String = head + """
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> s: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x; let t = id.y; let hd = id.z; let n = p.a.x; let heads = p.a.y; let kv = p.a.z; let dim = p.a.w;
  let pos0 = u32(p.b.y); let nk = pos0 + n;
  if (j >= nk) { return; }
  let at = (t * heads + hd) * nk + j;
  if (j > pos0 + t) { s[at] = -1e30; return; }
  let kvh = hd / (heads / kv);
  var acc = 0.0;
  for (var i = 0u; i < dim; i++) { acc += q[t * heads * dim + hd * dim + i] * k[j * kv * dim + kvh * dim + i]; }
  s[at] = acc * p.b.x;
}"""
  /** y[t, h, i] = softmax_j(s[t, h, j]) · v[j, kv(h), i] */
  val attend: String = head + """
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; let t = id.y; let hd = id.z; let n = p.a.x; let heads = p.a.y; let kv = p.a.z; let dim = p.a.w;
  if (i >= dim) { return; }
  let pos0 = u32(p.b.y); let nk = pos0 + n; let last = pos0 + t;
  let row = (t * heads + hd) * nk; let kvh = hd / (heads / kv);
  var m = -1e30;
  for (var j = 0u; j <= last; j++) { m = max(m, s[row + j]); }
  var z = 0.0;
  for (var j = 0u; j <= last; j++) { z += exp(s[row + j] - m); }
  var acc = 0.0;
  for (var j = 0u; j <= last; j++) { acc += exp(s[row + j] - m) / z * v[j * kv * dim + kvh * dim + i]; }
  y[t * heads * dim + hd * dim + i] = acc;
}"""
  private def elementwise(expr: String) = head + s"""
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.y * 4194240u + id.x; if (i >= p.a.x) { return; }
  let g = a[i]; let u = b[i];
  y[i] = $expr;
}"""
  val add: String = elementwise("g + u")
  val swiglu: String = elementwise("g / (1.0 + exp(-g)) * u")
