/**
 * uihtml reverse channel (HTML → MATLAB) for a finished run.
 *
 * When a script creates a `uihtml` component with a callback
 * (HTMLEventReceivedFcn / DataChangedFcn), the run leaves entries in
 * `rt.uihtmlCallbacks`. `executeCode` then builds a `UihtmlSession` and returns
 * it so the host (the worker) can keep the interpreter alive and re-enter it
 * when an event arrives from the iframe — mirroring MATLAB, where the script
 * returns but its callbacks keep firing.
 *
 * Dispatch re-activates the run's own special-builtin closures (captured at
 * end-of-run, NOT re-registered — re-registering would reset their counters)
 * and pushes the runtime so `disp`, plotting, and `sendEventToHTMLSource`
 * inside the callback reach this runtime. This mirrors executeCode's own
 * save/restore of SPECIAL_BUILTIN_NAMES.
 */
import type { Runtime } from "./runtime.js";
import type { PlotInstruction } from "../../graphics/types.js";
import { RTV } from "./constructors.js";
import { pushCurrentRuntime, popCurrentRuntime, decref } from "./refcount.js";
import { isRuntimeFunction, type RuntimeFunction } from "./types.js";
import {
  getIBuiltin,
  registerDynamicIBuiltin,
  type IBuiltin,
} from "../interpreter/builtins/types.js";
import { SPECIAL_BUILTIN_NAMES } from "./specialBuiltinNames.js";
import { convertJsonValue } from "../interpreter/builtins/misc.js";

export interface UihtmlSession {
  /** True while at least one component still has a registered callback. */
  hasCallbacks(): boolean;
  /** Dispatch an event from a component's page into MATLAB. `eventType` is
   *  "HTMLEventReceived" (JS `sendEventToMATLAB`) or "DataChanged" (JS set
   *  `htmlComponent.Data`). `payload.data` is the structured-clone'd JS value.
   *  New plot output is flushed via `onDrawnow`; outgoing
   *  `sendEventToHTMLSource` calls go through `rt.onHtmlSourceEvent`. */
  dispatchEvent(
    compId: string,
    eventType: "HTMLEventReceived" | "DataChanged",
    payload: { name?: string; data: unknown }
  ): void;
  /** Release the retained callback handles (and their refs). */
  dispose(): void;
}

/**
 * Build a session over a runtime that registered uihtml callbacks.
 * `activeSpecials` is a snapshot of this runtime's special-builtin closures,
 * captured at end-of-run while they are still installed globally.
 */
export function createUihtmlSession(
  rt: Runtime,
  activeSpecials: Map<string, IBuiltin>,
  onDrawnow?: (plotInstructions: PlotInstruction[]) => void
): UihtmlSession {
  function invokeHandle(fn: RuntimeFunction, args: unknown[]): void {
    if (fn.jsFn) {
      if (fn.jsFnExpectsNargout) fn.jsFn(0, ...args);
      else fn.jsFn(...args);
      return;
    }
    rt.dispatch(fn.name, 0, args);
  }

  function dispatchEvent(
    compId: string,
    eventType: "HTMLEventReceived" | "DataChanged",
    payload: { name?: string; data: unknown }
  ): void {
    const entry = rt.uihtmlCallbacks.get(compId);
    const fn = entry?.[eventType];
    if (!fn || !isRuntimeFunction(fn)) return;

    // Re-activate this runtime's special builtins for the re-entry, exactly as
    // executeCode does around a run: save current, install ours, restore after.
    const saved = new Map<string, IBuiltin>();
    for (const name of SPECIAL_BUILTIN_NAMES) {
      const ex = getIBuiltin(name);
      if (ex) saved.set(name, ex);
    }
    for (const ib of activeSpecials.values()) registerDynamicIBuiltin(ib);

    pushCurrentRuntime(rt);
    const before = rt.plotInstructions.length;
    try {
      const src = RTV.struct({ ComponentId: RTV.char(compId) });
      const dataRV = convertJsonValue(payload.data);
      const event =
        eventType === "HTMLEventReceived"
          ? RTV.struct({
              HTMLEventName: RTV.char(payload.name ?? ""),
              HTMLEventData: dataRV,
              Source: src,
              EventName: RTV.char("HTMLEventReceived"),
            })
          : RTV.struct({
              Data: dataRV,
              PreviousData: RTV.tensor(new Float64Array(0), [0, 0]),
              Source: src,
              EventName: RTV.char("DataChanged"),
            });
      invokeHandle(fn, [src, event]);
    } finally {
      popCurrentRuntime(rt);
      for (const ib of saved.values()) registerDynamicIBuiltin(ib);
    }

    const newInstrs = rt.plotInstructions.slice(before);
    if (newInstrs.length && onDrawnow) onDrawnow(newInstrs);
  }

  function dispose(): void {
    for (const entry of rt.uihtmlCallbacks.values()) {
      if (entry.HTMLEventReceived) decref(rt, entry.HTMLEventReceived);
      if (entry.DataChanged) decref(rt, entry.DataChanged);
    }
    rt.uihtmlCallbacks.clear();
  }

  return {
    hasCallbacks: () => rt.uihtmlCallbacks.size > 0,
    dispatchEvent,
    dispose,
  };
}
