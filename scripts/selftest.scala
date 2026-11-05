//> using scala 3.3.8
//> using dep com.microsoft.playwright:playwright:1.62.0

import com.microsoft.playwright.{BrowserType, Playwright}
import java.nio.file.Path
import scala.jdk.CollectionConverters.*

/** THE BROWSER ENGINE'S SELF-TEST in Chromium with WebGPU (`$CHROME`, else Playwright's chromium-1234): what
 * browser/index.html writes (#selftest), printed. The page served over http (WebGPU wants a secure context;
 * localhost is one): `scripts/selftest.sh` links, serves and runs it. `scala-cli run scripts/selftest.scala -- <url>` */
@main def selftest(url: String): Unit =
  val exe = sys.env.getOrElse("CHROME", sys.props("user.home") +
    "/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/" +
    "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing")
  val pw = Playwright.create(Playwright.CreateOptions().setEnv(Map("PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD" -> "1").asJava))
  val browser =
    if sys.env.get("BROWSER").contains("webkit") then pw.webkit().launch(BrowserType.LaunchOptions().setHeadless(false))
    else pw.chromium().launch(BrowserType.LaunchOptions().setExecutablePath(Path.of(exe)).setHeadless(false)
      .setArgs(List("--enable-unsafe-webgpu", "--enable-features=WebGPU").asJava))
  val p = browser.newPage()
  p.setDefaultTimeout(600000)
  val console = scala.collection.mutable.ArrayBuffer.empty[String]
  p.onConsoleMessage(m => console += m.text())
  p.onPageError(e => console += s"error: $e")
  p.navigate(url)
  val finished =
    try { p.waitForFunction("() => /selftest: (done|failed)/.test(document.body.textContent)"); true }
    catch case _: com.microsoft.playwright.TimeoutError => false
  val text = if finished then p.locator("#selftest").textContent()
    else "selftest: failed: no end in 10 minutes; the console:\n" + console.mkString("\n")
  println(text)
  browser.close(); pw.close()
  if text.contains("failed") then sys.exit(1)
