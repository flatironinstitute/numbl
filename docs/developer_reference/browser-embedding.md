# Browser embedding (`numbl/browser`)

Third-party web apps embed numbl at two levels.

## Managed sessions: `createNumblSession`

The batteries-included path, exported from the `numbl/browser` entry: numbl owns the worker, the VFS, the mip bootstrap, and persistence.

```ts
import { createNumblSession } from "numbl/browser";

const session = await createNumblSession({
  files: [{ path: "main.m", content: source }],
  mainFile: "main.m",
  onOutput: console.log,
  onHtmlSourceEvent: (compId, name, dataJson) => {
    /* script -> host */
  },
});
session.writeFile("data.bin", bytes); // into the session VFS
const out = await session.readFile("result.json"); // back out of the VFS
await session.dispatchHtmlEvent(compId, "go", payload); // host -> script
```

Sessions also run incrementally: `mainFile` is optional, and `session.execute(code)` runs code against a persistent workspace (REPL semantics: variables, globals, persistents, warning and PRNG state, and hold state carry across calls, also past a failed call, and expression results auto-display). Each call resolves with an `ExecuteResult` (`{ ok, output, plotInstructions, error?, aborted? }`); numbl errors resolve with `ok: false` and a formatted message rather than rejecting. This is the path notebook-style hosts use (e.g. the JupyterLite kernel).

```ts
const session = await createNumblSession({ onOutput: console.log });
await session.execute("x = linspace(0, 2*pi, 100);");
const { plotInstructions } = await session.execute("plot(x, sin(x))");
```

`readFile` lets a host run a script standalone and read back what it wrote, with no uihtml event bridge when the script has nothing interactive about it.

- **Worker inlining.** The session worker is bundled standalone at build time and embedded as text in the published entry; it is started from a Blob URL, so consumers need no bundler worker support.
- **Boot.** Restore the persisted `/system` directory, write project files under `/project/`, fetch mip core from its GitHub release if missing (same URL/proxy/cachebust behavior as the IDE; `mip: false` skips this), then run through an `ExecSession` with the mip directory on the search path. Scripts can then `mip load --install <pkg>`.
- **Persistence.** A dependency-free IndexedDB store holds `/system` (mip core + installed packages). After the run and after each dispatched event, the VFS change set filtered to `/system/` is written back, so installs survive page loads. The store is wiped after an inactivity period (default 30 min, configurable), so a rebuilt/updated package set refreshes without a manual cache clear.
- **uihtml bridge.** uihtml plot instructions surface as components; `sendEventToHTMLSource` arrives via a callback; `dispatchHtmlEvent` fires the script's `HTMLEventReceivedFcn` and resolves/rejects when the callback returns. The `UihtmlSession` stays live after the main script finishes, the same mechanism the IDE uses (see [uihtml.md](uihtml.md)).
- **Interrupt and input.** When the page is cross-origin isolated (`SharedArrayBuffer` available), `interrupt()` cancels the running `execute` cooperatively (it resolves with `aborted: true` and the workspace survives), and `input()` calls `onInputRequest`, answered with `provideInput(text)`. Both use the blocking channel ([web-app.md](web-app.md#blocking-waits)), so `interrupt()` also ends a blocked `input()`, `pause` or `gpu.call` at once. `canInterrupt` and `canInput` report whether this works; without isolation `input()` errors and a runaway run can only be stopped with `dispose()`.
- **Figures.** Sessions do not render figures; every plot instruction the run produces is returned (`execute`) for the host to render. The `numbl/graphics` entry provides the React renderer (`figuresReducer`/`FigureView`) plus `restoreNaNs` for instructions that crossed a JSON boundary. uihtml components additionally surface through the uihtml bridge above.

- **Capabilities.** The session worker builds its host with `bootBrowserHost` (`src/browserHost.ts`, shared with the IDE worker) and awaits `host.ready` at boot: it has the WebGPU helper (`gpu` in `.numbl.js`), but not qhull (delaunay/convhull) or the wasm LAPACK accelerator. The worker is an esbuild bundle started from a Blob URL, so it has no asset URL for `qhull.wasm`, and inlining it would add about 0.5 MB to every embedding page; the accelerator needs a hosting URL. Either could become a `createNumblSession` option, filled in through the same boot function.

## Raw primitives: the root export

For hosts that want their own worker: `executeCode` (synchronous, platform-agnostic), `ExecSession` (REPL-style state carried across `executeCode` calls), `createHost` (the worker's capabilities, passed as `ExecOptions.host` or `new ExecSession({ host })`; see [architecture.md](architecture.md#host-capabilities)), `SessionState` (the state one passes as `ExecOptions.session` to share it between `executeCode` calls), `VirtualFileSystem` (with change tracking), `BrowserFileIOAdapter` (sync-XHR `websave`/`webread` with GitHub-release CORS proxying, and `unzip`), `BrowserSystemAdapter`, and the `UihtmlSession` type. For Stop and blocking waits, a host passes `createCancelBuffer()` as `cancelSAB`, stops with `signalCancel`, and can serve `input()` through `blockingChannelFor(cancelSAB)` on the worker side and `createChannelServer` on its own thread. `executeCode` scans `searchPaths` directories for workspace files the same way a runtime `addpath` does.
