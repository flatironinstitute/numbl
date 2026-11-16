/**
 * `numbl plot-server`: a plot viewer driven by an external program.
 *
 * Starts the plot server immediately and keeps one `ExecSession` alive.
 * Clients POST a frame to `/eval` (JSON or binary, see `plotFrame.ts`): its
 * variables are put into the session's base workspace, its code is run, and
 * the resulting plot instructions go to the viewer. This lets a program in
 * another language (e.g. the C client) plot its arrays with MATLAB plot
 * commands.
 *
 * The response is JSON `{ ok: true, output }` or `{ ok: false, error, output }`.
 *
 * The server binds to 127.0.0.1 only: `/eval` runs arbitrary code. For a
 * remote simulation, use an ssh tunnel.
 */

import type { IncomingMessage, ServerResponse } from "node:http";
import { startPlotServer, type PlotServer } from "./cli-plot-server.js";
import { ExecSession } from "./numbl-core/execSession.js";
import type { NumblHost } from "./numbl-core/host.js";
import type { WorkspaceFile } from "./numbl-core/workspace/types.js";
import type { OptLevel } from "./numbl-core/executors/plugins.js";
import {
  diagnoseErrors,
  formatDiagnostics,
} from "./numbl-core/diagnostics/index.js";
import { NodeFileIOAdapter } from "./cli-fileio.js";
import { NodeSystemAdapter } from "./cli-system.js";
import { decodeFrame, type PlotFrame } from "./plotFrame.js";

const MAX_BODY_BYTES = 1024 * 1024 * 1024;

export interface PlotSessionOptions {
  port: number;
  workspaceFiles: WorkspaceFile[];
  searchPaths: string[];
  host: NumblHost;
  optimization: OptLevel;
}

export async function runPlotSession(opts: PlotSessionOptions): Promise<void> {
  const session = new ExecSession({
    workspaceFiles: [...opts.workspaceFiles],
    searchPaths: [...opts.searchPaths],
    host: opts.host,
  });
  const fileIO = new NodeFileIOAdapter();
  const system = new NodeSystemAdapter();
  // Assigned once the server is listening; requests only arrive after that.
  let server: PlotServer | undefined = undefined;

  const evalRequest = (body: Buffer): { status: number; reply: object } => {
    let frame: PlotFrame;
    try {
      frame = decodeFrame(body);
    } catch (e) {
      return { status: 400, reply: { ok: false, error: errMsg(e) } };
    }
    let output = "";
    const onText = (text: string) => {
      output += text;
    };
    try {
      const result = session.execute(
        frame.code,
        {
          displayResults: true,
          initialVariableValues: frame.vars,
          onOutput: onText,
          onErrorOutput: onText,
          onDrawnow: instrs => server?.sendInstructions(instrs),
          optimization: opts.optimization,
          fileIO,
          system,
        },
        "plot-server"
      );
      if (result.plotInstructions.length > 0) {
        server?.sendInstructions(result.plotInstructions);
      }
      return { status: 200, reply: { ok: true, output } };
    } catch (error) {
      const diags = diagnoseErrors(
        error,
        frame.code,
        "plot-server",
        session.workspaceFiles
      );
      return {
        status: 200,
        reply: { ok: false, error: formatDiagnostics(diags), output },
      };
    }
  };

  const handleRequest = (
    req: IncomingMessage,
    res: ServerResponse,
    url: URL
  ): boolean => {
    if (url.pathname !== "/eval") return false;
    if (req.method !== "POST") {
      res.writeHead(405);
      res.end();
      return true;
    }
    readBody(req, MAX_BODY_BYTES, async body => {
      const { status, reply } =
        body === null
          ? { status: 413, reply: { ok: false, error: "request too large" } }
          : evalRequest(body);
      // Reply only once the viewer has taken the plot data, so a client that
      // sends faster than the browser draws is slowed down (and its own
      // latest-frame-wins queue drops the frames in between).
      await server?.waitForDrain();
      const json = JSON.stringify(reply);
      res.writeHead(status, {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(json),
      });
      res.end(json);
    });
    return true;
  };

  server = await startPlotServer({
    port: opts.port,
    host: "127.0.0.1",
    handleRequest,
  });
  // Machine-readable line for a client that spawned this process.
  process.stdout.write(`NUMBL_PLOT_URL=http://127.0.0.1:${server.port}/\n`);
  await server.closed;
}

/** Collect the request body; `null` when it exceeds `limit`. */
function readBody(
  req: IncomingMessage,
  limit: number,
  done: (body: Buffer | null) => void | Promise<void>
): void {
  const chunks: Buffer[] = [];
  let size = 0;
  req.on("data", (chunk: Buffer) => {
    size += chunk.length;
    if (size <= limit) chunks.push(chunk);
  });
  req.on("end", () => {
    void done(size > limit ? null : Buffer.concat(chunks));
  });
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
