// okay-llm: language models as effects (okay-scala), on no hardware in particular — the model an effect (`Llm`)
// answered by a module of its own (okay-mlx: Apple's MLX; others alike), a gateway speaking OpenAI's and Anthropic's
// APIs, the programs ask and bench. JVM and Scala Native
import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

ThisBuild / scalaVersion := "3.3.8"
ThisBuild / organization := "io.github.okay-effects"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Wunused:imports,privates,locals")
ThisBuild / resolvers += "okay-scala" at "https://okay-effects.github.io/okay-scala/maven"
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.txt"))
ThisBuild / homepage := Some(url("https://github.com/okay-effects/okay-llm"))

/** THE LIBRARY: the effects, the APIs' formats, the gateway, the server, the programs — and a fake model */
lazy val core = crossProject(JVMPlatform, NativePlatform).crossType(CrossType.Pure).in(file("core"))
  .settings(name := "okay-llm",
    libraryDependencies ++= Seq("io.github.okay-effects" %%% "okay-scala" % "0.1.1",
      "org.scalameta" %%% "munit" % "1.2.3" % Test))

lazy val root = (project in file(".")).aggregate(core.componentProjects.map(p => p: ProjectReference): _*)
  .settings(publish / skip := true)
