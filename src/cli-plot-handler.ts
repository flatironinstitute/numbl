/**
 * Lazy plot-server handler used by numbl's CLI and reused by sibling
 * projects (mtoc2) that drive the same plot pipeline from a different
 * execution backend.
 *
 * The pattern:
 *   - On the first batch of `PlotInstruction`s, start the plot server
 *     (which serves the static plot-viewer SPA + opens it in a
 *     browser, then streams instructions over SSE).
 *   - Buffer instructions arriving while the server is still starting
 *     so none are dropped (the first call's promise wraps the boot).
 *   - On `flushAndWait(remaining)`, push the trailing batch and signal
 *     `scriptDone()` so the viewer's status bar reads "complete".
 *     The server stays alive until the user hits Ctrl+C — that's how
 *     numbl's CLI lets you keep looking at the plot after the script
 *     exits.
 *
 * The `disabled` flag short-circuits the whole thing: when true,
 * `onDrawnow` is `undefined` (the caller drops instructions) and
 * `flushAndWait` is a no-op. Used by the `--no-plot` / non-`--plot`
 * CLI paths and the cross-runner harness, neither of which wants a
 * browser tab to pop open during script execution.
 */

import {
  startPlotServer,
  type PlotServer,
  type PlotServerOptions,
} from "./cli-plot-server.js";
import type { PlotInstruction } from "./graphics/types.js";
import type { UihtmlSession } from "./numbl-core/executeCode.js";

export interface PlotHandler {
  /** Called per batch of instructions (typically by the runtime's
   *  `onDrawnow` callback, but any producer can call it). `undefined`
   *  when `disabled` was passed — callers should skip producing
   *  instructions entirely in that case. */
  onDrawnow: ((instructions: PlotInstruction[]) => void) | undefined;
  /** Flush a trailing batch and wait for the server to close (i.e.,
   *  for the user to Ctrl+C). Always safe to call; no-op when the
   *  server never started. */
  flushAndWait: (remaining?: PlotInstruction[]) => Promise<void>;
  /** Push a uihtml event to the viewer (interpreter → page). Use as the
   *  runtime's `onHtmlSourceEvent` hook. No-op when disabled / no server. */
  sendUihtmlEvent: (compId: string, name: string, dataJson: string) => void;
  /** Set the live session so events POSTed back from the viewer re-enter the
   *  interpreter. Call after the run with `result.uihtmlSession`. */
  setUihtmlSession: (session: UihtmlSession | null) => void;
}

export function createPlotHandler(
  disabled: boolean,
  plotOpts?: PlotServerOptions
): PlotHandler {
  if (disabled) {
    return {
      onDrawnow: undefined,
      flushAndWait: async () => {},
      sendUihtmlEvent: () => {},
      setUihtmlSession: () => {},
    };
  }

  let plotServer: PlotServer | null = null;
  let serverStarting: Promise<PlotServer> | null = null;
  const pendingBatches: PlotInstruction[][] = [];
  let session: UihtmlSession | null = null;

  // Wire viewer-posted events (page → interpreter) into the live session.
  const wireServer = (ps: PlotServer) => {
    ps.setUihtmlEventHandler(e => {
      if (!session) return;
      const eventType =
        e.kind === "dataChanged" ? "DataChanged" : "HTMLEventReceived";
      try {
        session.dispatchEvent(e.compId, eventType, {
          name: e.name,
          data: e.data,
        });
      } catch (err) {
        process.stderr.write(
          `Error in uihtml callback: ${
            err instanceof Error ? err.message : String(err)
          }\n`
        );
      }
    });
  };

  const onDrawnow = (instructions: PlotInstruction[]) => {
    if (plotServer) {
      plotServer.sendInstructions(instructions);
    } else {
      pendingBatches.push(instructions);
      if (!serverStarting) {
        serverStarting = startPlotServer(plotOpts).then(ps => {
          plotServer = ps;
          wireServer(ps);
          for (const batch of pendingBatches) {
            ps.sendInstructions(batch);
          }
          pendingBatches.length = 0;
          return ps;
        });
      }
    }
  };

  const flushAndWait = async (remaining: PlotInstruction[] = []) => {
    if (remaining.length > 0) {
      onDrawnow(remaining);
    }
    if (serverStarting) {
      plotServer = await serverStarting;
    }
    if (plotServer) {
      plotServer.scriptDone();
      await plotServer.closed;
    }
  };

  const sendUihtmlEvent = (compId: string, name: string, dataJson: string) => {
    plotServer?.sendUihtmlEvent(compId, name, dataJson);
  };

  const setUihtmlSession = (s: UihtmlSession | null) => {
    session = s;
  };

  return { onDrawnow, flushAndWait, sendUihtmlEvent, setUihtmlSession };
}
