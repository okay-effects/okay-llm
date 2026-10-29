package okay.llm

import okay.effects.json.{Descent, Json}

/** okay-simple's JSON (okay-effects-json: an ordered `Obj`, its recursive-descent parser) read the way an API's body
 * is read here: a field or `Null`, a value at its type */
extension (j: Json)
  def apply(key: String): Json = j match
    case o: Json.Obj => o.get(key).getOrElse(Json.Null)
    case _ => Json.Null
  def get(key: String): Option[Json] = j match
    case o: Json.Obj => o.get(key).filter(_ != Json.Null)
    case _ => None
  def str: String = j match { case Json.Str(s) => s; case other => sys.error(s"not text: ${Json.kind(other)}") }
  def int: Int = j match { case Json.Num(n) => n.toInt; case other => sys.error(s"not a number: ${Json.kind(other)}") }
  def bool: Boolean = j match
    case Json.Bool(b) => b
    case other => sys.error(s"not a boolean: ${Json.kind(other)}")
  def items: Vector[Json] = j match
    case Json.Arr(xs) => xs
    case other => sys.error(s"not a list: ${Json.kind(other)}")
  def fields: Vector[(String, Json)] = j match
    case Json.Obj(f) => f
    case other => sys.error(s"not a record: ${Json.kind(other)}")
  /** compact JSON text */
  def render: String = Json.print(j)

/** a text as JSON; a malformed one an `IllegalArgumentException`, saying where and what was expected */
def parseJson(text: String): Json = Descent.parse(text).fold(e => throw IllegalArgumentException(e.toString), identity)

/** a string as a JSON string literal */
def quoteJson(s: String): String = Json.print(Json.Str(s))
