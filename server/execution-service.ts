import { createServer, type IncomingMessage, type ServerResponse } from "http";
import { spawn } from "child_process";
import { mkdtemp, mkdir, writeFile, rm } from "fs/promises";
import { dirname, join } from "path";
import { tmpdir } from "os";

interface ServerOptions {
  port: number;
  numblCliPath: string;
  passkey: string;
}

interface ProjectFile {
  name: string;
  content: string;
}

interface ExecutionRequest {
  files: ProjectFile[];
  mainScript: string;
  optimization?: number;
  fuse?: boolean;
}

const MAX_BODY_BYTES = 10 * 1024 * 1024; // 10 MB
const DEFAULT_MAX_CONCURRENT = 3;
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_MEMORY_MB = 1024;

let activeExecutions = 0;
let nativeAddonAvailable: boolean | null = null;

async function probeNativeAddon(numblCliPath: string): Promise<boolean> {
  return new Promise(resolve => {
    let output = "";
    const child = spawn("node", [numblCliPath, "info"], {
      timeout: 5000,
    });
    child.stdout?.on("data", d => {
      output += d.toString();
    });
    child.on("close", code => {
      if (code === 0) {
        try {
          const info = JSON.parse(output.trim());
          resolve(!!info.nativeAddon);
          return;
        } catch {
          // parse failed
        }
      }
      resolve(false);
    });
    child.on("error", () => resolve(false));
  });
}

function setCorsHeaders(res: ServerResponse): void {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

function sendJson(
  res: ServerResponse,
  status: number,
  data: Record<string, unknown>
): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("Request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

async function handleExecute(
  req: IncomingMessage,
  res: ServerResponse,
  numblCliPath: string
): Promise<void> {
  const maxConcurrent =
    parseInt(process.env.NUMBL_MAX_CONCURRENT || "") || DEFAULT_MAX_CONCURRENT;
  const timeoutMs =
    parseInt(process.env.NUMBL_TIMEOUT_MS || "") || DEFAULT_TIMEOUT_MS;
  const maxMemoryMB =
    parseInt(process.env.NUMBL_MAX_MEMORY_MB || "") || DEFAULT_MAX_MEMORY_MB;

  if (activeExecutions >= maxConcurrent) {
    sendJson(res, 503, {
      success: false,
      error: "Server is at maximum capacity. Please try again later.",
    });
    return;
  }

  let body: string;
  try {
    body = await readBody(req);
  } catch {
    sendJson(res, 400, {
      success: false,
      error: "Failed to read request body",
    });
    return;
  }

  let parsed: ExecutionRequest;
  try {
    parsed = JSON.parse(body);
  } catch {
    sendJson(res, 400, { success: false, error: "Invalid JSON" });
    return;
  }

  const { files, mainScript, optimization, fuse } = parsed;

  if (!files || !Array.isArray(files) || files.length === 0) {
    sendJson(res, 400, {
      success: false,
      error: "Invalid request: 'files' must be a non-empty array",
    });
    return;
  }
  if (!mainScript || typeof mainScript !== "string") {
    sendJson(res, 400, {
      success: false,
      error: "Invalid request: 'mainScript' must be a string",
    });
    return;
  }
  if (!files.some(f => f.name === mainScript)) {
    sendJson(res, 400, {
      success: false,
      error: `Main script '${mainScript}' not found in project files`,
    });
    return;
  }

  activeExecutions++;
  let tempDir: string | null = null;

  try {
    tempDir = await mkdtemp(join(tmpdir(), "numbl-exec-"));

    // Create subdirectories and write all files
    const dirs = new Set<string>();
    for (const file of files) {
      const filePath = join(tempDir!, file.name);
      const dir = dirname(filePath);
      if (dir !== tempDir && !dirs.has(dir)) {
        await mkdir(dir, { recursive: true });
        dirs.add(dir);
      }
    }
    await Promise.all(
      files.map(file =>
        writeFile(join(tempDir!, file.name), file.content, "utf-8")
      )
    );

    // SSE headers
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });

    const sendEvent = (data: Record<string, unknown>) => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    const _tempDir = tempDir;
    await new Promise<void>((resolve, reject) => {
      const scriptPath = join(_tempDir, mainScript);
      let timedOut = false;
      let stdoutBuffer = "";

      const cliArgs = [numblCliPath, "run", "--stream"];
      if (optimization !== undefined) {
        cliArgs.push("--opt", String(optimization));
      }
      if (fuse) {
        cliArgs.push("--fuse");
      }
      cliArgs.push(scriptPath);
      const child = spawn("node", cliArgs, {
        cwd: _tempDir,
        env: {
          ...process.env,
          NODE_OPTIONS: `--max-old-space-size=${maxMemoryMB}`,
        },
      });

      req.on("close", () => {
        if (!child.killed) child.kill("SIGTERM");
      });

      const timeout = setTimeout(() => {
        timedOut = true;
        child.kill("SIGTERM");
        setTimeout(() => {
          if (!child.killed) child.kill("SIGKILL");
        }, 1000);
      }, timeoutMs);

      child.stdout?.on("data", (data: Buffer) => {
        stdoutBuffer += data.toString();
        const lines = stdoutBuffer.split("\n");
        stdoutBuffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            sendEvent(msg);
          } catch {
            sendEvent({ type: "output", text: line + "\n" });
          }
        }
      });

      child.stderr?.on("data", (data: Buffer) => {
        sendEvent({ type: "output", text: data.toString() });
      });

      child.on("close", code => {
        clearTimeout(timeout);
        if (stdoutBuffer.trim()) {
          try {
            const msg = JSON.parse(stdoutBuffer);
            sendEvent(msg);
          } catch {
            sendEvent({ type: "output", text: stdoutBuffer });
          }
        }
        if (timedOut) {
          sendEvent({
            type: "error",
            message: `Execution timed out after ${timeoutMs}ms`,
            errorType: "runtime",
            timedOut: true,
          });
        }
        sendEvent({ type: "close", success: !timedOut && code === 0 });
        res.end();
        resolve();
      });

      child.on("error", error => {
        clearTimeout(timeout);
        sendEvent({
          type: "error",
          message: `Failed to start process: ${error.message}`,
          errorType: "runtime",
        });
        sendEvent({ type: "close", success: false });
        res.end();
        reject(error);
      });
    });
  } catch (error) {
    if (!res.headersSent) {
      console.error("Execution error:", error);
      sendJson(res, 500, {
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown execution error",
      });
    }
  } finally {
    if (tempDir) {
      try {
        await rm(tempDir, { recursive: true, force: true });
      } catch (cleanupError) {
        console.error("Failed to clean up temp directory:", cleanupError);
      }
    }
    activeExecutions--;
  }
}

export function startServer(options: ServerOptions): void {
  const { port, numblCliPath, passkey } = options;

  const server = createServer(async (req, res) => {
    setCorsHeaders(res);

    // Handle CORS preflight
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    // Validate passkey on all endpoints
    const auth = req.headers["authorization"] || "";
    if (auth !== `Bearer ${passkey}`) {
      sendJson(res, 401, { error: "Unauthorized: invalid passkey" });
      return;
    }

    const url = new URL(req.url || "/", `http://localhost:${port}`);

    if (url.pathname === "/health" && req.method === "GET") {
      sendJson(res, 200, {
        status: "ok",
        activeExecutions,
        nativeAddon: nativeAddonAvailable,
      });
      return;
    }

    if (url.pathname === "/execute" && req.method === "POST") {
      await handleExecute(req, res, numblCliPath);
      return;
    }

    sendJson(res, 404, { error: "Not found" });
  });

  server.listen(port, async () => {
    console.log(`numbl execution server listening on http://localhost:${port}`);
    nativeAddonAvailable = await probeNativeAddon(numblCliPath);
    console.log(
      `Native addon: ${nativeAddonAvailable ? "available" : "not available"}`
    );
  });
}
