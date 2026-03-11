#!/usr/bin/env node

import express from "express";
import { spawn } from "child_process";
import { mkdtemp, writeFile, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import cors from "cors";

// Configuration
interface ServiceConfig {
  port: number;
  maxConcurrentExecutions: number;
  executionTimeoutMs: number;
  maxMemoryMB: number;
}

const DEFAULT_CONFIG: ServiceConfig = {
  port: 3001,
  maxConcurrentExecutions: 3,
  executionTimeoutMs: 30000, // 30 seconds
  maxMemoryMB: 1024, // 1 GB
};

// Load config from environment or use defaults
const config: ServiceConfig = {
  port: parseInt(process.env.NUMBL_SERVICE_PORT || "") || DEFAULT_CONFIG.port,
  maxConcurrentExecutions:
    parseInt(process.env.NUMBL_MAX_CONCURRENT || "") ||
    DEFAULT_CONFIG.maxConcurrentExecutions,
  executionTimeoutMs:
    parseInt(process.env.NUMBL_TIMEOUT_MS || "") ||
    DEFAULT_CONFIG.executionTimeoutMs,
  maxMemoryMB:
    parseInt(process.env.NUMBL_MAX_MEMORY_MB || "") ||
    DEFAULT_CONFIG.maxMemoryMB,
};

// Secret token for admin endpoints (required to be set for /update to work)
const UPDATE_SECRET = process.env.NUMBL_UPDATE_SECRET || "";

// Track active executions
let activeExecutions = 0;

// Probe the CLI at startup to check native addon availability
let nativeAddonAvailable: boolean | null = null;

async function probeNativeAddon(): Promise<boolean> {
  const numblPath = join(process.cwd(), "dist-cli", "cli.js");
  return new Promise(resolve => {
    let output = "";
    const child = spawn("node", [numblPath, "--info"], {
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

interface ProjectFile {
  name: string;
  content: string;
}

interface ExecutionRequest {
  files: ProjectFile[];
  mainScript: string;
}

const ALLOWED_ORIGINS = [
  "http://localhost:5173",
  "http://localhost:5174",
  "https://numbl.org",
];

const app = express();

// Middleware
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (e.g. curl, server-to-server)
      if (!origin) return callback(null, true);
      if (ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
      callback(new Error(`CORS: origin not allowed: ${origin}`));
    },
  })
);
app.use(express.json({ limit: "10mb" }));

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    activeExecutions,
    maxConcurrentExecutions: config.maxConcurrentExecutions,
    nativeAddon: nativeAddonAvailable,
  });
});

// Get configuration endpoint
app.get("/config", (req, res) => {
  res.json(config);
});

// Execute script endpoint (SSE streaming)
app.post("/execute", async (req, res) => {
  // Check concurrent execution limit
  if (activeExecutions >= config.maxConcurrentExecutions) {
    return res.status(503).json({
      success: false,
      error: "Server is at maximum capacity. Please try again later.",
    });
  }

  const { files, mainScript } = req.body as ExecutionRequest;

  // Validate request
  if (!files || !Array.isArray(files) || files.length === 0) {
    return res.status(400).json({
      success: false,
      error: "Invalid request: 'files' must be a non-empty array",
    });
  }

  if (!mainScript || typeof mainScript !== "string") {
    return res.status(400).json({
      success: false,
      error: "Invalid request: 'mainScript' must be a string",
    });
  }

  // Check if mainScript exists in files
  if (!files.some(f => f.name === mainScript)) {
    return res.status(400).json({
      success: false,
      error: `Main script '${mainScript}' not found in project files`,
    });
  }

  activeExecutions++;
  let tempDir: string | null = null;

  try {
    // Create temporary directory
    tempDir = await mkdtemp(join(tmpdir(), "numbl-exec-"));

    // Write all files to temp directory
    await Promise.all(
      files.map(file =>
        writeFile(join(tempDir!, file.name), file.content, "utf-8")
      )
    );

    // Set up SSE headers
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });

    const sendEvent = (data: Record<string, unknown>) => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    // Stream the script execution
    const _tempDir = tempDir;
    await new Promise<void>((resolve, reject) => {
      const scriptPath = join(_tempDir, mainScript);
      const numblPath = join(process.cwd(), "dist-cli", "cli.js");
      let timedOut = false;
      let stdoutBuffer = "";

      const child = spawn("node", [numblPath, "--stream", scriptPath], {
        cwd: _tempDir,
        env: {
          ...process.env,
          NODE_OPTIONS: `--max-old-space-size=${config.maxMemoryMB}`,
        },
      });

      // Kill child process if client disconnects
      req.on("close", () => {
        if (!child.killed) {
          child.kill("SIGTERM");
        }
      });

      // Set up timeout
      const timeout = setTimeout(() => {
        timedOut = true;
        child.kill("SIGTERM");
        setTimeout(() => {
          if (!child.killed) {
            child.kill("SIGKILL");
          }
        }, 1000);
      }, config.executionTimeoutMs);

      // Parse NDJSON lines from CLI stdout and forward as SSE events
      child.stdout?.on("data", (data: Buffer) => {
        stdoutBuffer += data.toString();
        const lines = stdoutBuffer.split("\n");
        // Keep the last (possibly incomplete) line in the buffer
        stdoutBuffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            sendEvent(msg);
          } catch {
            // Not valid JSON — send as plain output
            sendEvent({ type: "output", text: line + "\n" });
          }
        }
      });

      // Forward stderr as output events
      child.stderr?.on("data", (data: Buffer) => {
        sendEvent({ type: "output", text: data.toString() });
      });

      child.on("close", code => {
        clearTimeout(timeout);
        // Flush any remaining buffer
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
            message: `Execution timed out after ${config.executionTimeoutMs}ms`,
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
    // If headers haven't been sent yet, send JSON error
    if (!res.headersSent) {
      console.error("Execution error:", error);
      res.status(500).json({
        success: false,
        error:
          error instanceof Error ? error.message : "Unknown execution error",
      });
    }
  } finally {
    // Clean up temporary directory
    if (tempDir) {
      try {
        await rm(tempDir, { recursive: true, force: true });
      } catch (cleanupError) {
        console.error("Failed to clean up temp directory:", cleanupError);
      }
    }
    activeExecutions--;
  }
});

// Update endpoint — pulls latest code and rebuilds CLI and addon
app.post("/update", async (req, res) => {
  if (!UPDATE_SECRET) {
    return res.status(503).json({
      success: false,
      error: "Update endpoint is disabled (NUMBL_UPDATE_SECRET not set)",
    });
  }
  const authHeader = req.headers["authorization"] || "";
  if (authHeader !== `Bearer ${UPDATE_SECRET}`) {
    return res.status(401).json({ success: false, error: "Unauthorized" });
  }

  const projectDir = process.cwd();

  try {
    const output = await runCommands(
      [
        ["git", ["pull"]],
        ["npm", ["run", "build:addon"]],
        ["npm", ["run", "build:cli"]],
      ],
      projectDir
    );
    res.json({ success: true, output });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    res.status(500).json({ success: false, error: msg });
  }
});

async function runCommands(
  commands: [string, string[]][],
  cwd: string
): Promise<string> {
  let fullOutput = "";

  for (const [cmd, args] of commands) {
    fullOutput += `\n$ ${cmd} ${args.join(" ")}\n`;
    const result = await new Promise<string>((resolve, reject) => {
      let output = "";
      const child = spawn(cmd, args, { cwd });

      child.stdout?.on("data", d => {
        output += d.toString();
      });
      child.stderr?.on("data", d => {
        output += d.toString();
      });

      child.on("close", code => {
        if (code === 0) resolve(output);
        else
          reject(
            new Error(
              `Command '${cmd} ${args.join(" ")}' exited with code ${code}\n${output}`
            )
          );
      });

      child.on("error", err => reject(err));
    });
    fullOutput += result;
  }

  return fullOutput;
}

// Start the server
app.listen(config.port, async () => {
  console.log(`Numbl execution service running on port ${config.port}`);
  console.log(`Configuration:`, config);

  nativeAddonAvailable = await probeNativeAddon();
  console.log(
    `Native addon: ${nativeAddonAvailable ? "available" : "not available"}`
  );
});
