import { createServer, type ServerResponse } from "node:http";
import { exec } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";
import type { PlotInstruction } from "./graphics/types.js";

export interface PlotServer {
  /** Send a batch of plot instructions to all connected SSE clients */
  sendInstructions(instructions: PlotInstruction[]): void;
  /** Signal that the script has finished. Server stays alive until Ctrl+C. */
  scriptDone(): void;
  /** The port the server is listening on */
  readonly port: number;
  /** Promise that resolves when the server shuts down */
  readonly closed: Promise<void>;
}

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".json": "application/json",
};

export interface PlotServerOptions {
  /** Port to listen on (0 = OS-assigned free port) */
  port?: number;
  /** Address to bind to (default: "127.0.0.1") */
  host?: string;
}

export async function startPlotServer(
  options?: PlotServerOptions
): Promise<PlotServer> {
  // Resolve the dist-plot-viewer directory relative to this file
  const thisDir = fileURLToPath(new URL(".", import.meta.url));
  const distDir = join(thisDir, "..", "dist-plot-viewer");

  if (!existsSync(distDir)) {
    throw new Error(
      `Plot viewer not built. Run: npm run build:plot-viewer\n` +
        `Expected directory: ${distDir}`
    );
  }

  const sseClients: ServerResponse[] = [];
  let messageId = 0;
  const pendingMessages: string[] = [];

  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost`);

    // SSE endpoint
    if (url.pathname === "/events") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      res.write("\n"); // flush headers
      sseClients.push(res);

      // Flush buffered messages to this new client
      for (const msg of pendingMessages) {
        res.write(msg);
      }
      pendingMessages.length = 0;

      req.on("close", () => {
        const idx = sseClients.indexOf(res);
        if (idx >= 0) sseClients.splice(idx, 1);
      });
      return;
    }

    // Serve static files from dist-plot-viewer
    let filePath = url.pathname === "/" ? "/index.html" : url.pathname;
    const fullPath = join(distDir, filePath);

    // Security: ensure we stay within distDir
    if (!fullPath.startsWith(distDir)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }

    if (!existsSync(fullPath)) {
      // SPA fallback — serve index.html for unknown paths
      filePath = "/index.html";
    }

    const resolvedPath =
      filePath === "/index.html" ? join(distDir, "index.html") : fullPath;

    try {
      const content = readFileSync(resolvedPath);
      const ext = extname(resolvedPath);
      const contentType = MIME_TYPES[ext] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end("Not found");
    }
  });

  const listenPort = options?.port ?? 0;
  const listenHost = options?.host ?? "127.0.0.1";

  await new Promise<void>(resolve => {
    server.listen(listenPort, listenHost, () => resolve());
  });

  const port = (server.address() as { port: number }).port;
  const url = `http://${listenHost}:${port}/`;

  console.error(`[numbl] Plot server at ${url}`);
  openBrowser(url);

  let closedResolve: () => void;
  const closed = new Promise<void>(resolve => {
    closedResolve = resolve;
  });

  // Shut down cleanly on process exit
  const onExit = () => {
    // Destroy all open SSE connections so server.close() can complete
    for (const client of sseClients) {
      client.destroy();
    }
    sseClients.length = 0;
    server.close(() => closedResolve!());
  };
  process.on("SIGINT", onExit);
  process.on("SIGTERM", onExit);

  function sendInstructions(instructions: PlotInstruction[]): void {
    messageId++;
    const payload = `id: ${messageId}\ndata: ${JSON.stringify(instructions)}\n\n`;
    if (sseClients.length === 0) {
      pendingMessages.push(payload);
    } else {
      for (const client of sseClients) {
        client.write(payload);
      }
    }
  }

  function scriptDone(): void {
    messageId++;
    const payload = `id: ${messageId}\nevent: done\ndata: {}\n\n`;
    if (sseClients.length === 0) {
      pendingMessages.push(payload);
    } else {
      for (const client of sseClients) {
        client.write(payload);
      }
    }
    // Server stays alive until Ctrl+C (SIGINT handler above)
  }

  return { sendInstructions, scriptDone, port, closed };
}

function openBrowser(url: string): void {
  const platform = process.platform;
  let cmd: string;
  if (platform === "darwin") {
    cmd = `open "${url}"`;
  } else if (platform === "win32") {
    cmd = `start "" "${url}"`;
  } else {
    cmd = `xdg-open "${url}"`;
  }
  exec(cmd, err => {
    if (err) {
      console.error(`[numbl] Could not open browser. Visit: ${url}`);
    }
  });
}
