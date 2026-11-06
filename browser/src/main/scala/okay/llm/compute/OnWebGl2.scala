package okay.llm.compute

import scala.concurrent.Future
import scala.scalajs.js
import scala.scalajs.js.Dynamic.global as g
import scala.scalajs.js.typedarray.*

/**
 * THE DEVICE ON WEBGL2, where WebGPU is missing (Safari before 26, old phones): a tensor is a float texture of
 * four values a texel, its texels in a line 2048 wide (texel i at (i mod 2048, i div 2048)), a weight's words an
 * unsigned-integer texture the same way, its scales and biases float ones; an op a fragment shader drawn into the
 * output's texture (`Glsl`), a texel four outputs. Reads back at once (readPixels). Needs EXT_color_buffer_float.
 */
final class OnWebGl2 private (gl: js.Dynamic, val adapter: String) extends Compute:
  val api = "WebGL2"
  final class T(val rows: Int, val cols: Int, val tex: js.Dynamic, val h: Int)
  final class W(val out: Int, val in: Int, val words: js.Dynamic, val scales: js.Dynamic, val biases: js.Dynamic)
  private val WIDTH = 2048
  private val TEX = 0x0DE1; private val RGBA32F = 0x8814; private val RGBA = 0x1908; private val FLOAT = 0x1406
  private val RGBA32UI = 0x8D70; private val RGBA_INTEGER = 0x8D99; private val UINT = 0x1405
  private val NEAREST = 0x2600; private val FB = 0x8D40; private val COLOR0 = 0x8CE0
  /** the bytes held in textures now */
  var bytes = 0L

  private def height(texels: Int) = math.max(1, (texels + WIDTH - 1) / WIDTH)
  private def texture(h: Int, data: js.Any, floats: Boolean): js.Dynamic =
    val t = gl.createTexture()
    gl.bindTexture(TEX, t)
    if floats then gl.texImage2D(TEX, 0, RGBA32F, WIDTH, h, 0, RGBA, FLOAT, data)
    else gl.texImage2D(TEX, 0, RGBA32UI, WIDTH, h, 0, RGBA_INTEGER, UINT, data)
    gl.texParameteri(TEX, 0x2801, NEAREST); gl.texParameteri(TEX, 0x2800, NEAREST)
    gl.texParameteri(TEX, 0x2802, 0x812F); gl.texParameteri(TEX, 0x2803, 0x812F)
    bytes += WIDTH * h * 16L
    t
  private def floatTexture(data: Array[Float]): (js.Dynamic, Int) =
    val h = height((data.length + 3) / 4)
    val padded = new Float32Array(WIDTH * h * 4); padded.set(data.toTypedArray)
    (texture(h, padded, floats = true), h)
  private def wordTexture(data: Array[Int]): js.Dynamic =
    val h = height((data.length + 3) / 4)
    val padded = new Uint32Array(WIDTH * h * 4); padded.set(new Uint32Array(data.toTypedArray.buffer))
    texture(h, padded, floats = false)
  def tensor(rows: Int, cols: Int, data: Array[Float]): T =
    require(cols % 4 == 0, s"cols $cols not a multiple of 4")
    val (t, h) = floatTexture(data); T(rows, cols, t, h)
  def quantized(q: QLinear): W =
    W(q.out, q.in, wordTexture(q.words), floatTexture(q.scales)._1, floatTexture(q.biases)._1)
  // A POOL OF TEXTURES by height: an op takes one of its size, a release puts it back — a texture created and
  // deleted for every op (thousands a question) was freed late by a phone's driver and the tab died; the pool holds
  // at most `pooled` bytes, the rest deleted
  private val pool = scala.collection.mutable.Map.empty[Int, List[js.Dynamic]]
  private var pooledBytes = 0L
  private val pooled = 96L * 1048576
  private def empty(rows: Int, cols: Int): T =
    val h = height(rows * cols / 4)
    pool.getOrElse(h, Nil) match
      case t :: rest => pool(h) = rest; pooledBytes -= WIDTH * h * 16L; T(rows, cols, t, h)
      case Nil => T(rows, cols, texture(h, null, floats = true), h)
  def release(x: T): Unit =
    if pooledBytes + WIDTH * x.h * 16L <= pooled then { pool(x.h) = x.tex :: pool.getOrElse(x.h, Nil); pooledBytes += WIDTH * x.h * 16L }
    else { gl.deleteTexture(x.tex); bytes -= WIDTH * x.h * 16L }

  private def program(code: String): js.Dynamic =
    def shader(kind: Int, src: String) =
      val s = gl.createShader(kind); gl.shaderSource(s, src); gl.compileShader(s)
      if !gl.getShaderParameter(s, 0x8B81).asInstanceOf[Boolean] then
        throw IllegalStateException("a shader: " + gl.getShaderInfoLog(s))
      s
    val p = gl.createProgram()
    gl.attachShader(p, shader(0x8B31, Glsl.vertex)); gl.attachShader(p, shader(0x8B30, code)); gl.linkProgram(p)
    if !gl.getProgramParameter(p, 0x8B82).asInstanceOf[Boolean] then
      throw IllegalStateException("a program: " + gl.getProgramInfoLog(p))
    p
  private lazy val linearP = program(Glsl.linear); private lazy val normP = program(Glsl.rmsNorm)
  private lazy val ropeP = program(Glsl.headNormRope); private lazy val scoresP = program(Glsl.scores)
  private lazy val attendP = program(Glsl.attend); private lazy val addP = program(Glsl.add)
  private lazy val swigluP = program(Glsl.swiglu); private lazy val sliceP = program(Glsl.slice)
  private lazy val concatP = program(Glsl.concat)
  private val vao = gl.createVertexArray()
  private val fb = gl.createFramebuffer()

  /** one draw into `out`: the program, its int and float uniforms, its textures by name */
  private def draw(p: js.Dynamic, out: T, ints: Seq[(String, Int)], floats: Seq[(String, Float)],
      textures: Seq[(String, js.Dynamic)]): T =
    gl.bindFramebuffer(FB, fb)
    gl.framebufferTexture2D(FB, COLOR0, TEX, out.tex, 0)
    gl.viewport(0, 0, WIDTH, out.h)
    gl.useProgram(p)
    gl.uniform1i(gl.getUniformLocation(p, "total"), out.rows * out.cols / 4)
    for (name, v) <- ints do gl.uniform1i(gl.getUniformLocation(p, name), v)
    for (name, v) <- floats do gl.uniform1f(gl.getUniformLocation(p, name), v)
    for ((name, tex), unit) <- textures.zipWithIndex do
      gl.activeTexture(0x84C0 + unit); gl.bindTexture(TEX, tex)
      gl.uniform1i(gl.getUniformLocation(p, name), unit)
    gl.bindVertexArray(vao)
    gl.drawArrays(0x0004, 0, 3)
    for unit <- textures.indices do { gl.activeTexture(0x84C0 + unit); gl.bindTexture(TEX, null) }
    out

  def linear(x: T, w: W): T =
    draw(linearP, empty(x.rows, w.out), Seq("rows" -> x.rows, "inn" -> w.in, "outn" -> w.out), Nil,
      Seq("X" -> x.tex, "Wd" -> w.words, "S" -> w.scales, "B" -> w.biases))
  def rmsNorm(x: T, w: T, eps: Float): T =
    draw(normP, empty(x.rows, x.cols), Seq("cols" -> x.cols), Seq("eps" -> eps), Seq("X" -> x.tex, "Wn" -> w.tex))
  def headNormRope(x: T, w: T, heads: Int, headDim: Int, eps: Float, theta: Float, pos0: Int): T =
    draw(ropeP, empty(x.rows, x.cols), Seq("heads" -> heads, "dim" -> headDim, "pos0" -> pos0), Seq("eps" -> eps, "theta" -> theta),
      Seq("X" -> x.tex, "Wn" -> w.tex))
  def attention(q: T, k: T, v: T, heads: Int, kvHeads: Int, headDim: Int, pos0: Int): T =
    val n = q.rows; val nk = k.rows; val np = (nk + 3) / 4 * 4
    require(nk == pos0 + n, s"k has $nk rows for $pos0 + $n")
    val ints = Seq("n" -> n, "nk" -> nk, "np" -> np, "heads" -> heads, "kv" -> kvHeads, "dim" -> headDim, "pos0" -> pos0)
    val s = draw(scoresP, empty(n * heads, np), ints, Seq("scale" -> (1.0 / math.sqrt(headDim)).toFloat),
      Seq("Q" -> q.tex, "K" -> k.tex))
    val out = draw(attendP, empty(n, heads * headDim), ints, Nil, Seq("S" -> s.tex, "V" -> v.tex))
    release(s)
    out
  def add(a: T, b: T): T = draw(addP, empty(a.rows, a.cols), Nil, Nil, Seq("A" -> a.tex, "B" -> b.tex))
  def swiglu(gate: T, up: T): T =
    draw(swigluP, empty(gate.rows, gate.cols), Nil, Nil, Seq("A" -> gate.tex, "B" -> up.tex))
  def concat(a: T, b: T): T =
    draw(concatP, empty(a.rows + b.rows, a.cols), Seq("na" -> (a.rows * a.cols / 4)), Nil, Seq("A" -> a.tex, "B" -> b.tex))
  def slice(x: T, from: Int, until: Int): T =
    draw(sliceP, empty(until - from, x.cols), Seq("off" -> from * x.cols / 4), Nil, Seq("X" -> x.tex))
  def rows(x: T, at: Seq[Int]): Future[Vector[Array[Float]]] =
    gl.bindFramebuffer(FB, fb)
    gl.framebufferTexture2D(FB, COLOR0, TEX, x.tex, 0)
    val all = new Float32Array(WIDTH * x.h * 4)
    gl.readPixels(0, 0, WIDTH, x.h, RGBA, FLOAT, all)
    Future.successful(at.toVector.map(r => all.subarray(r * x.cols, (r + 1) * x.cols).toArray))

object OnWebGl2:
  /** the device on WebGL2 — or why not: no WebGL2, no float render targets, textures too small, a shader refused */
  def create(): Either[String, OnWebGl2] =
    try
      val canvas = g.document.createElement("canvas")
      val gl = canvas.getContext("webgl2")
      if gl == null || js.isUndefined(gl) then Left("this browser has no WebGL2 either")
      else if gl.getExtension("EXT_color_buffer_float") == null then
        Left("WebGL2 here has no float render targets (EXT_color_buffer_float)")
      else if gl.getParameter(0x0D33).asInstanceOf[Int] < 4096 then Left("WebGL2's textures are too small here")
      else
        val info = gl.getExtension("WEBGL_debug_renderer_info")
        val name = if info == null then gl.getParameter(0x1F01).toString
          else gl.getParameter(info.UNMASKED_RENDERER_WEBGL).toString
        Right(OnWebGl2(gl, name))
    catch case e: Throwable => Left(s"WebGL2 failed: ${e.getMessage}")

/** THE SHADERS, in GLSL ES 3.00: a fragment a texel of the output, four values; `total` the output's texels */
object Glsl:
  val vertex: String = "#version 300 es\nvoid main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); " +
    "gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }"
  private val head = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp usampler2D;
out vec4 o;
uniform int total;
ivec2 at(int i) { return ivec2(i & 2047, i >> 11); }
vec4 tex4(sampler2D s, int i) { return texelFetch(s, at(i), 0); }
float tex1(sampler2D s, int i) { return texelFetch(s, at(i >> 2), 0)[i & 3]; }
uint word(usampler2D s, int i) { return texelFetch(s, at(i >> 2), 0)[i & 3]; }
int idx() { ivec2 c = ivec2(gl_FragCoord.xy); return c.y * 2048 + c.x; }
"""
  val linear: String = head + """
uniform sampler2D X; uniform usampler2D Wd; uniform sampler2D S; uniform sampler2D B;
uniform int rows; uniform int inn; uniform int outn;
float dotRow(int oo, int r) {
  int groups = inn / 64; int perRow = inn / 8; int in4 = inn / 4;
  float a = 0.0;
  for (int gi = 0; gi < groups; gi++) {
    float s = tex1(S, oo * groups + gi); float b = tex1(B, oo * groups + gi);
    for (int k = 0; k < 8; k++) {
      uint w = word(Wd, oo * perRow + gi * 8 + k); int xb = r * in4 + gi * 16 + k * 2;
      vec4 x0 = tex4(X, xb); vec4 x1 = tex4(X, xb + 1);
      vec4 q0 = vec4(float(w & 15u), float((w >> 4u) & 15u), float((w >> 8u) & 15u), float((w >> 12u) & 15u));
      vec4 q1 = vec4(float((w >> 16u) & 15u), float((w >> 20u) & 15u), float((w >> 24u) & 15u), float(w >> 28u));
      a += dot(s * q0 + b, x0) + dot(s * q1 + b, x1);
    }
  }
  return a;
}
void main() {
  int i = idx(); if (i >= total) discard;
  int o4 = outn / 4; int r = i / o4; int oc = (i - r * o4) * 4;
  o = vec4(dotRow(oc, r), dotRow(oc + 1, r), dotRow(oc + 2, r), dotRow(oc + 3, r));
}"""
  val rmsNorm: String = head + """
uniform sampler2D X; uniform sampler2D Wn; uniform int cols; uniform float eps;
void main() {
  int i = idx(); if (i >= total) discard;
  int c4 = cols / 4; int r = i / c4; int c = i - r * c4;
  float ss = 0.0;
  for (int k = 0; k < c4; k++) { vec4 v = tex4(X, r * c4 + k); ss += dot(v, v); }
  o = tex4(X, i) * inversesqrt(ss / float(cols) + eps) * tex4(Wn, c);
}"""
  val headNormRope: String = head + """
uniform sampler2D X; uniform sampler2D Wn; uniform int heads; uniform int dim; uniform float eps; uniform float theta;
uniform int pos0;
float ro(int r, int hd, int d, float s) {
  int hf = dim / 2; int base = r * heads * dim + hd * dim;
  int lo = d < hf ? d : d - hf;
  float x1 = tex1(X, base + lo) * s * tex1(Wn, lo); float x2 = tex1(X, base + lo + hf) * s * tex1(Wn, lo + hf);
  float a = float(pos0 + r) * pow(theta, -2.0 * float(lo) / float(dim));
  return d < hf ? x1 * cos(a) - x2 * sin(a) : x2 * cos(a) + x1 * sin(a);
}
void main() {
  int i = idx(); if (i >= total) discard;
  int c4 = heads * dim / 4; int r = i / c4; int c = (i - r * c4) * 4; int hd = c / dim; int d = c - hd * dim;
  int base = r * c4 + hd * dim / 4; int h4 = dim / 4;
  float ss = 0.0;
  for (int k = 0; k < h4; k++) { vec4 v = tex4(X, base + k); ss += dot(v, v); }
  float s = inversesqrt(ss / float(dim) + eps);
  o = vec4(ro(r, hd, d, s), ro(r, hd, d + 1, s), ro(r, hd, d + 2, s), ro(r, hd, d + 3, s));
}"""
  /** the scores [n · heads, np]: s[t, h, j] = q[t, h] · k[j, kv(h)] scaled, -1e30 beyond the causal edge */
  val scores: String = head + """
uniform sampler2D Q; uniform sampler2D K; uniform int n; uniform int nk; uniform int np; uniform int heads;
uniform int kv; uniform int dim; uniform float scale; uniform int pos0;
float score(int t, int hd, int j) {
  if (j > pos0 + t || j >= nk) return -1e30;
  int kvh = hd / (heads / kv); int qb = t * heads * dim / 4 + hd * dim / 4; int kb = j * kv * dim / 4 + kvh * dim / 4;
  float a = 0.0;
  for (int k = 0; k < dim / 4; k++) a += dot(tex4(Q, qb + k), tex4(K, kb + k));
  return a * scale;
}
void main() {
  int i = idx(); if (i >= total) discard;
  int c4 = np / 4; int row = i / c4; int j = (i - row * c4) * 4; int t = row / heads; int hd = row - t * heads;
  o = vec4(score(t, hd, j), score(t, hd, j + 1), score(t, hd, j + 2), score(t, hd, j + 3));
}"""
  /** y[t, h, i..i+3] = softmax_j(s[t, h, j]) · v[j, kv(h), i..i+3] */
  val attend: String = head + """
uniform sampler2D S; uniform sampler2D V; uniform int n; uniform int nk; uniform int np; uniform int heads;
uniform int kv; uniform int dim; uniform int pos0;
void main() {
  int i = idx(); if (i >= total) discard;
  int c4 = heads * dim / 4; int t = i / c4; int c = (i - t * c4) * 4; int hd = c / dim; int d = c - hd * dim;
  int kvh = hd / (heads / kv); int row = (t * heads + hd) * np; int last = pos0 + t;
  float m = -1e30;
  for (int j = 0; j <= last; j++) m = max(m, tex1(S, row + j));
  float z = 0.0;
  for (int j = 0; j <= last; j++) z += exp(tex1(S, row + j) - m);
  vec4 acc = vec4(0.0);
  for (int j = 0; j <= last; j++) acc += exp(tex1(S, row + j) - m) / z * tex4(V, j * kv * dim / 4 + kvh * dim / 4 + d / 4);
  o = acc;
}"""
  private def elementwise(expr: String) = head + s"""
uniform sampler2D A; uniform sampler2D B;
void main() {
  int i = idx(); if (i >= total) discard;
  vec4 g = tex4(A, i); vec4 u = tex4(B, i);
  o = $expr;
}"""
  val add: String = elementwise("g + u")
  val swiglu: String = elementwise("g / (1.0 + exp(-g)) * u")
  val slice: String = head + """
uniform sampler2D X; uniform int off;
void main() { int i = idx(); if (i >= total) discard; o = tex4(X, i + off); }"""
  /** the rows of A (na texels) then the rows of B */
  val concat: String = head + """
uniform sampler2D A; uniform sampler2D B; uniform int na;
void main() { int i = idx(); if (i >= total) discard; o = i < na ? tex4(A, i) : tex4(B, i - na); }"""
