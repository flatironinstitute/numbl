# uihtml: interactive HTML components

`uihtml` renders a self-contained HTML document in a figure and gives it a
two-way bridge to the interpreter, targeting MATLAB's standard `uihtml` API so
the same `.m` runs in numbl and real MATLAB. This file describes the internals;
for authoring figures see [../uihtml-figures.md](../uihtml-figures.md).

## Render path

`uihtml` is a `.m` shim (a `handle` classdef) that parses name-value options and
calls a native primitive, which emits a `uihtml` `PlotInstruction` carrying a
stable component **id**, the HTML markup string, and (if given) the `Data`
already `jsonencode`d. Like all plotting, nothing renders in numbl-core — the
instruction flows through the normal `onDrawnow` path to the viewer, which
renders it as an `<iframe srcdoc>` instead of the axes/trace canvas.

The viewer wraps the page in a small **bootstrap** before serving the srcdoc.
The bootstrap supplies the standard `htmlComponent` JavaScript object
(`Data` get/set, `addEventListener`, `sendEventToMATLAB`), calls the page's
`setup(htmlComponent)`, and — for incoming data — `JSON.parse`s it onto
`htmlComponent.Data` and fires the page's `"DataChanged"` listeners. The
component id is injected so messages can be routed.

## Data: script → page

Setting `Data` on the shim `jsonencode`s it into the instruction; the bootstrap
parses it and pushes it to the page. Because numbl renders at construction (and
on `show`), `Data` is supplied at construction or refreshed via `show(h)` — numbl
does not hook a post-construction property set.

## Reverse channel: page → script

This is the key subsystem. Because a numbl run normally finishes and discards its
`Runtime`, but the figure stays interactive afterward, the reverse channel needs
the interpreter to remain callable.

**Keeping the runtime alive.** When a run registers a uihtml callback
(`HTMLEventReceivedFcn` / `DataChangedFcn`, recorded on the runtime), the run
returns a **session** over the still-live runtime instead of letting it be
collected. The host (the browser worker) retains the session so later iframe
events can re-enter the interpreter. Registered handles are refcount-incref'd so
they (and any captured environment — closures, captured handle objects) survive
the run; this is why anonymous functions and captures work unchanged.

**Lifetime.** Exactly one session is armed at a time. A session is armed only if
the run left a callback (otherwise zero overhead). It is disposed — decref'ing
the handles — when a new run/REPL command starts or figures are cleared. This
mirrors the IDE clearing figures at the start of every run, so a new run disarms
the previous one.

**Dispatch.** An event in the page posts a message to the host (tagged with the
component id and kind); the host forwards it to the worker, which calls
`session.dispatchEvent`. Dispatch temporarily re-activates this runtime's
special-builtin closures (snapshotted at end-of-run — re-registering them would
reset their internal counters), pushes the runtime, builds the MATLAB event
struct (`HTMLEventName`/`HTMLEventData` or `Data`/`PreviousData`, plus a `src`
struct carrying the component id), and invokes the handle the same way `feval`
does. New plot output produced by the callback is flushed through `onDrawnow`.

**Outgoing (`sendEventToHTMLSource`).** A special builtin reads the component id
from `src`, `jsonencode`s the data, and calls a runtime hook; the host relays it
to the matching iframe, where the bootstrap delivers it to the page's
`addEventListener(name)` listeners.

Re-entrancy is naturally serialized: the worker is single-threaded, so a run, a
callback, and a blocking `input()` cannot overlap; events queue until idle.

## Surfaces

- **Browser IDE** — full reverse channel (the worker holds the session;
  `window` message relays connect iframe ↔ worker).
- **MATLAB** — native `uihtml`; the portable authoring rules keep one `.m`
  working in both.
- **CLI `--plot`** — renders and accepts data, but has no reverse channel: the
  process keeps no live interpreter after the run, and the transport is one-way
  (server-sent events). Adding it would need a client→server event endpoint and
  a retained runtime.
