/**
 * numbl/browser — managed in-browser sessions.
 *
 * Usage:
 *   import { createNumblSession } from "numbl/browser";
 *   const session = await createNumblSession({
 *     files: [{ path: "main.m", content: "mip load --install chebfun;\n..." }],
 *     mainFile: "main.m",
 *     onOutput: console.log,
 *   });
 *   session.dispatchHtmlEvent(compId, "eventName", payload);
 *
 * The session runs in a worker numbl manages (inlined at build time — no
 * bundler worker support needed), with mip bootstrapped from its GitHub
 * release and installed packages persisted in IndexedDB across page loads.
 *
 * Not available in sessions: delaunay/convhull (qhull WASM is not loaded)
 * and non-uihtml figures (plot instructions other than uihtml are ignored).
 */
export {
  createNumblSession,
  type NumblSession,
  type NumblSessionOptions,
} from "./session.js";
export type { BootFile, UihtmlComponent } from "./protocol.js";
