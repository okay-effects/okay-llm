package okay.llm.read

/**
 * A MODEL ANSWERING `Read` IN PLACE (okay-desk specs/data.md §15; calculus.md §16): the JVM's handler of the effect
 * — one method, the operation answered. There is no second list of what a skill needs: `Read`'s operations are that
 * list; a backend (mlx-scala's forward, the page's own program on a `Compute` device) answers them. `Read.sync`
 * makes it a handler, `Read.run` runs a program on it.
 */
trait Reading:
  def answer[X](op: Read[X]): X
