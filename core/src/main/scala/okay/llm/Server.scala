package okay.llm

import okay.effects.json.Json

import java.io.{BufferedInputStream, InputStream, OutputStream}
import java.net.{InetAddress, ServerSocket}

/** THE SERVER: HTTP/1.1 on a socket (the JVM's and Scala Native's java.net alike) — a connection a request, served
 * one at a time (one model, one GPU), each to the program `serve` makes of it */
object Server:
  def run(port: Int, host: String = "127.0.0.1")(serve: (Request, OutputStream) => Unit): Unit =
    val server = ServerSocket(port, 16, InetAddress.getByName(host))
    System.err.println(s"listening on http://$host:$port")
    while true do
      val socket = server.accept()
      try
        val in = BufferedInputStream(socket.getInputStream)
        val out = socket.getOutputStream
        read(in).foreach { req =>
          try serve(req, out)
          catch case e: Throwable =>
            System.err.println(s"${req.method} ${req.path}: $e")
            val body = s"""{"error":{"type":"api_error","message":${quoteJson(e.toString)}}}"""
            try out.write((s"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n" +
              s"Content-Length: ${body.getBytes("UTF-8").length}\r\nConnection: close\r\n\r\n$body").getBytes("UTF-8"))
            catch case _: java.io.IOException => ()
        }
      catch case e: java.io.IOException => System.err.println(s"connection: $e")
      finally socket.close()

  /** a request: its line, headers, and body — by Content-Length, or chunked */
  def read(in: InputStream): Option[Request] =
    val first = line(in)
    if first.isEmpty then return None
    val Array(method, path, _*) = first.split(' '): @unchecked
    val headers = Iterator.continually(line(in)).takeWhile(_.nonEmpty).flatMap { h =>
      val i = h.indexOf(':')
      if i < 0 then None else Some(h.substring(0, i).trim.toLowerCase -> h.substring(i + 1).trim)
    }.toMap
    val body =
      if headers.get("transfer-encoding").exists(_.toLowerCase.contains("chunked")) then
        val b = java.io.ByteArrayOutputStream()
        var size = Integer.parseInt(line(in).takeWhile(_ != ';').trim, 16)
        while size > 0 do
          b.write(in.readNBytes(size))
          line(in)
          size = Integer.parseInt(line(in).takeWhile(_ != ';').trim, 16)
        b.toByteArray
      else in.readNBytes(headers.get("content-length").map(_.toInt).getOrElse(0))
    Some(Request(method, path, headers, String(body, "UTF-8")))

  /** a line to CRLF (or LF), without it */
  private def line(in: InputStream): String =
    val b = java.io.ByteArrayOutputStream()
    var c = in.read()
    while c >= 0 && c != '\n' do
      if c != '\r' then b.write(c)
      c = in.read()
    String(b.toByteArray, "UTF-8")
