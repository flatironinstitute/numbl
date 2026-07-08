# Browser embedding (`numbl/browser`)

Third-party web apps embed numbl at two levels.

## Managed sessions — `createNumblSession`

The batteries-included path, exported from the `numbl/browser` entry: numbl
owns the worker, the VFS, the mip bootstrap, and persistence.

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

`readFile` lets a host run a script standalone and read back what it wrote —
no uihtml event bridge needed when the script has nothing interactive about
it.

- **Worker inlining.** The session worker is bundled standalone at build
  time and embedded as text in the published entry; it is started from a
  Blob URL, so consumers need no bundler worker support.
- **Boot.** Restore the persisted `/system` directory, write project files
  under `/project/`, fetch mip core from its GitHub release if missing
  (same URL/proxy/cachebust behavior as the IDE), then `executeCode` with
  the mip directory on the search path. Scripts can then
  `mip load --install <pkg>`.
- **Persistence.** A dependency-free IndexedDB store holds `/system` (mip
  core + installed packages). After the run and after each dispatched
  event, the VFS change set filtered to `/system/` is written back, so
  installs survive page loads. The store is wiped after an inactivity
  period (default 30 min, configurable), so a rebuilt/updated package set
  refreshes without a manual cache clear.
- **uihtml bridge.** uihtml plot instructions surface as components;
  `sendEventToHTMLSource` arrives via a callback; `dispatchHtmlEvent` fires
  the script's `HTMLEventReceivedFcn` and resolves/rejects when the
  callback returns. The `UihtmlSession` stays live after the main script
  finishes — same mechanism the IDE uses (see [uihtml.md](uihtml.md)).

Not available in sessions: qhull-backed builtins (delaunay/convhull — the
WASM is not loaded) and non-uihtml figures (other plot instructions are
ignored; embedding hosts render their own UI).

## Raw primitives — the root export

For hosts that want their own worker: `executeCode` (synchronous,
platform-agnostic), `VirtualFileSystem` (with change tracking),
`BrowserFileIOAdapter` (sync-XHR `websave`/`webread` with GitHub-release
CORS proxying, and `unzip`), `BrowserSystemAdapter`, and the
`UihtmlSession` type. `executeCode` scans `searchPaths` directories for
workspace files the same way a runtime `addpath` does.
