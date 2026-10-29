// okay-llm: language models as effects (okay-scala), on no hardware in particular — the model an effect (`Llm`)
// answered by a module of its own (okay-mlx: Apple's MLX; others alike), a gateway speaking OpenAI's and Anthropic's
// APIs, the programs ask and bench. JVM and Scala Native
import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

ThisBuild / scalaVersion := "3.3.8"
ThisBuild / organization := "io.github.okay-effects"
ThisBuild / version := "0.1.1-SNAPSHOT"
ThisBuild / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Wunused:imports,privates,locals")
ThisBuild / resolvers ++= Seq("okay-scala" at "https://okay-effects.github.io/okay-scala/maven",
  "okay-simple" at "https://okay-effects.github.io/okay-simple/maven")
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt"))
ThisBuild / homepage := Some(url("https://github.com/okay-effects/okay-llm"))

ThisBuild / developers := List(Developer("sergey-scherbina", "Sergiy Shcherbyna", "sergey.scherbina@gmail.com",
  url("https://github.com/sergey-scherbina")))
ThisBuild / scmInfo := Some(ScmInfo(url("https://github.com/okay-effects/okay-llm"),
  "scm:git@github.com:okay-effects/okay-llm.git"))
ThisBuild / versionScheme := Some("early-semver")

/** RELEASES, ON GITHUB PAGES: `sbt publish` writes the Maven repository into target/pages-maven, which goes to the
 * gh-pages branch under maven/ — https://okay-effects.github.io/okay-llm/maven */
val pagesRepository = settingKey[File]("the Maven repository a release is written to, for GitHub Pages")
ThisBuild / pagesRepository := (ThisBuild / baseDirectory).value / "target" / "pages-maven"
ThisBuild / publishTo := Some(MavenCache("pages", (ThisBuild / pagesRepository).value))
ThisBuild / Compile / packageDoc / publishArtifact := false

/** THE LIBRARY: the effects, the APIs' formats, the gateway, the server, the programs — and a fake model */
lazy val core = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Pure).in(file("core"))
  .settings(name := "okay-llm",
    libraryDependencies ++= Seq("io.github.okay-effects" %%% "okay-scala" % "0.1.1",
      "io.github.okay-effects" %%% "okay-effects-json-value" % "0.1.1",
      "org.scalameta" %%% "munit" % "1.2.3" % Test))

lazy val root = (project in file(".")).aggregate(core.componentProjects.map(p => p: ProjectReference): _*)
  .settings(publish / skip := true)
