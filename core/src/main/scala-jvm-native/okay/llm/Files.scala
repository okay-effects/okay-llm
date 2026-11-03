package okay.llm

import java.nio.file.{Files as NioFiles, Path}
import okay.scala3.*

/** FILES: a text read */
enum Files[X]:
  case Read(path: Path) extends Files[String]

object Files:
  def read[R[_]](path: Path)(using In[Files, R]): String ! R = perform[Files](Read(path))

  def disk[R[_], Rest[_], A](p: A ! R)(using Split[Files, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Files, Rest, A, A](pure)([X] => (op: Files[X]) => op match
      case Read(path) => NioFiles.readString(path)))

  /** files given: a map of paths to texts */
  def of[R[_], Rest[_], A](files: Map[Path, String])(p: A ! R)(using Split[Files, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Files, Rest, A, A](pure)([X] => (op: Files[X]) => op match
      case Read(path) => files.getOrElse(path, throw java.io.FileNotFoundException(path.toString))))
