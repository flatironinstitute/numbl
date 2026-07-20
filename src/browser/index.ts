/**
 * numbl/browser — managed in-browser sessions.
 *
 * Run-a-script usage:
 *   import { createNumblSession } from "numbl/browser";
 *   const session = await createNumblSession({
 *     files: [{ path: "main.m", content: "mip load --install chebfun;\n..." }],
 *     mainFile: "main.m",
 *     onOutput: console.log,
 *   });
 *   session.dispatchHtmlEvent(compId, "eventName", payload);
 *
 * REPL usage (persistent workspace across calls):
 *   const session = await createNumblSession({ onOutput: console.log });
 *   await session.execute("x = linspace(0, 2*pi, 100);");
 *   const { plotInstructions } = await session.execute("plot(x, sin(x))");
 *   // render plotInstructions with figuresReducer/FigureView from numbl/graphics
 *
 * The session runs in a worker numbl manages (inlined at build time — no
 * bundler worker support needed), with mip bootstrapped from its GitHub
 * release and installed packages persisted in IndexedDB across page loads.
 *
 * Figures are not rendered by the session itself: plot instructions are
 * returned/surfaced for the host to render (numbl/graphics provides a React
 * renderer). Not available in sessions: delaunay/convhull (qhull WASM is not
 * loaded).
 */
export {
  createNumblSession,
  type NumblSession,
  type NumblSessionOptions,
} from "./session.js";
export type { BootFile, ExecuteResult, UihtmlComponent } from "./protocol.js";
export type { PlotInstruction } from "../graphics/types.js";
