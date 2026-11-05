package okay.llm.qwen

import scala.scalajs.js

/** UNICODE NORMALIZATION FORM C in the browser: the string's own `normalize` (the JVM's `Nfc` has the tables) */
object Nfc:
  def apply(s: String): String = s.asInstanceOf[js.Dynamic].normalize("NFC").asInstanceOf[String]
