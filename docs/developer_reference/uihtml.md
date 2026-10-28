# uihtml: interactive HTML components

`uihtml` renders a self-contained HTML document in a figure and gives it a two-way bridge to the interpreter, targeting MATLAB's standard `uihtml` API so the same `.m` runs in numbl and real MATLAB. This file describes the internals; for authoring figures see [../uihtml-figures.md](../uihtml-figures.md).

## Render path

`uihtml` is a `.m` shim (`src/stdlib/shims/uihtml.m`, a `handle` classdef) that parses name-value options and calls the native primitive `drawuihtml` (`plotting/builtins.ts`), which emits a `uihtml` `PlotInstruction` carrying a stable component **id**, the HTML markup string, and (if given) the `Data` already `jsonencode`d. Like all plotting, nothing renders in numbl-core: the instruction flows through the normal `onDrawnow` path to the viewer, which renders it as an `<iframe srcdoc>` instead of the axes/trace canvas.

The viewer wraps the page in a small **bootstrap** (`src/graphics/uihtmlSrcDoc.ts`) before serving the srcdoc. The bootstrap supplies the standard `htmlComponent` JavaScript object (`Data` get/set, `addEventListener`, `sendEventToMATLAB`), calls the page's `setup(htmlComponent)`, and, for incoming data, `JSON.parse`s it onto `htmlComponent.Data` and fires the page's `"DataChanged"` listeners. The component id is injected so messages can be routed.

## Data: script → page

Setting `Data` on the shim `jsonencode`s it into the instruction; the bootstrap parses it and pushes it to the page. Because numbl renders at construction (and on `show`), `Data` is supplied at construction or refreshed via `show(h)`; numbl does not hook a post-construction property set.

## Reverse channel: page → script

This is the key subsystem. Because a numbl run normally finishes and discards its `Runtime`, but the figure stays interactive afterward, the reverse channel needs the interpreter to remain callable.

**Keeping the runtime alive.** When a run registers a uihtml callback (`HTMLEventReceivedFcn` / `DataChangedFcn`, recorded in `rt.uihtmlCallbacks` by `registeruihtmlcallback`), `executeCode` returns a `UihtmlSession` (`runtime/uihtmlSession.ts`) over the still-live runtime instead of letting it be collected. The host (the browser worker's `ExecSession`, or the CLI plot handler) retains the session so later iframe events can re-enter the interpreter. Registered handles are refcount-incref'd so they (and any captured environment, such as closures and captured handle objects) survive the run; this is why anonymous functions and captures work unchanged.

**Lifetime.** Exactly one session is armed at a time. A session is armed only if the run left a callback (otherwise zero overhead). It is disposed (decref'ing the handles) when a new run or REPL command starts or the workspace is cleared. This mirrors the IDE clearing figures at the start of every run, so a new run disarms the previous one.

**Dispatch.** An event in the page posts a message to the host (tagged with the component id and kind); the host forwards it to the worker, which calls `session.dispatchEvent`. Dispatch pushes the runtime, builds the MATLAB event struct (`HTMLEventName`/`HTMLEventData` or `Data`/`PreviousData`, plus `Source`, a struct carrying the `ComponentId`, which is also passed as `src`), and invokes the handle. New plot output produced by the callback is flushed through `onDrawnow`.

**Outgoing (`sendEventToHTMLSource`).** A builtin reads the component id from `src`, `jsonencode`s the data, and calls the runtime hook `rt.onHtmlSourceEvent`; the host relays it to the matching iframe, where the bootstrap delivers it to the page's `addEventListener(name)` listeners.

Re-entrancy is naturally serialized: the worker is single-threaded, so a run, a callback, and a blocking `input()` cannot overlap; events queue until idle.

## Surfaces

The session machinery above is transport-agnostic; each surface differs only in how iframe events reach the runtime and how outgoing events reach the iframe.

- **Browser IDE**: interpreter (worker) and figure (iframe) share a page, so the relay is `postMessage` (`src/graphics/uihtmlRelay.ts`): the IDE forwards the iframe's `window` message to the worker, and forwards the worker's outgoing event to the iframe.
- **CLI `numbl run --plot`**: interpreter (Node) and figure (a browser tab) are separate processes bridged over HTTP. The plot server (`src/cli-plot-server.ts`) has a `POST /uihtml-event` endpoint (page → interpreter, dispatched into the held session) and a `uihtml` server-sent event (interpreter → page, relayed to the iframe). The CLI already stays alive after the run to keep the figure open, which is exactly the window in which events are dispatched. Only `run --plot` is wired; the CLI REPL is not.
- **MATLAB**: native `uihtml`; the portable authoring rules keep one `.m` working across all of these.
