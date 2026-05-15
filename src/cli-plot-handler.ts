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
}

export function createPlotHandler(
  disabled: boolean,
  plotOpts?: PlotServerOptions
): PlotHandler {
  if (disabled) {
    return {
      onDrawnow: undefined,
      flushAndWait: async () => {},
    };
  }

  let plotServer: PlotServer | null = null;
  let serverStarting: Promise<PlotServer> | null = null;
  const pendingBatches: PlotInstruction[][] = [];

  const onDrawnow = (instructions: PlotInstruction[]) => {
    if (plotServer) {
      plotServer.sendInstructions(instructions);
    } else {
      pendingBatches.push(instructions);
      if (!serverStarting) {
        serverStarting = startPlotServer(plotOpts).then(ps => {
          plotServer = ps;
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

  return { onDrawnow, flushAndWait };
}
