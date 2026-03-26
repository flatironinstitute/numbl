import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

interface BrowserWasmManifestTarget {
  name: string;
  wasmPath: string;
  exports?: string[];
}

interface BrowserWasmManifest {
  targets?: BrowserWasmManifestTarget[];
}

function createImportObject(): WebAssembly.Imports {
  return {
    wasi_snapshot_preview1: {
      fd_write: () => 0,
      fd_read: () => 0,
      fd_close: () => 0,
      fd_seek: () => 0,
      fd_fdstat_get: () => 0,
      proc_exit: () => {},
      environ_sizes_get: () => 0,
      environ_get: () => 0,
      clock_time_get: () => 0,
      args_sizes_get: () => 0,
      args_get: () => 0,
    },
    env: {
      emscripten_notify_memory_growth: () => {},
    },
  };
}

function resolveBuiltWasmPath(repoRoot: string, wasmPath: string): string {
  const relative = wasmPath.replace(/^\//, "");
  return join(repoRoot, "public", relative);
}

describe("browser Wasm artifacts", () => {
  const repoRoot = process.cwd();
  const manifestPath = join(repoRoot, "public", "wasm-kernels", "manifest.json");
  const maybeIt = existsSync(manifestPath) ? it : it.skip;

  maybeIt("instantiate and expose the manifest-declared exports", async () => {
    const manifest = JSON.parse(
      readFileSync(manifestPath, "utf8")
    ) as BrowserWasmManifest;
    const targets = manifest.targets ?? [];

    expect(targets.length).toBeGreaterThan(0);

    for (const target of targets) {
      const wasmFile = resolveBuiltWasmPath(repoRoot, target.wasmPath);
      expect(existsSync(wasmFile), `${target.name}: missing ${wasmFile}`).toBe(true);

      const bytes = readFileSync(wasmFile);
      const { instance } = await WebAssembly.instantiate(
        bytes,
        createImportObject()
      );
      const exports = instance.exports as Record<string, unknown>;

      for (const exportName of target.exports ?? []) {
        expect(
          exports[exportName],
          `${target.name}: missing export ${exportName}`
        ).toBeDefined();
      }
    }
  });
});
