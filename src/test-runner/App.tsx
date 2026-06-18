import { Fragment, useEffect, useState } from "react";
import { executeCode } from "../numbl-core/executeCode.js";
import type { WorkspaceFile } from "../numbl-core/workspace/types.js";
import { VirtualFileSystem } from "../vfs/VirtualFileSystem.js";
import { BrowserFileIOAdapter } from "../vfs/BrowserFileIOAdapter.js";
import { BrowserSystemAdapter } from "../vfs/BrowserSystemAdapter.js";
import { ensureQhullBackend } from "../numbl-core/native/qhull-browser.js";

const GITHUB_BASE =
  "https://github.com/flatironinstitute/numbl/blob/main/numbl_test_scripts";

interface ManifestEntry {
  path: string;
  workspace: string[];
  skip?: string;
}
interface Manifest {
  tests: ManifestEntry[];
  allFiles: string[];
  sources: Record<string, string>;
  binaries: Record<string, string>;
}

// Absolute VFS prefix for test files. addpath tests resolve relative paths
// against the VFS cwd (default /project via BrowserFileIOAdapter), so
// matching that prefix keeps scanDirectory() lookups consistent.
const VFS_ROOT = "/project";
const TEXT_ENCODER = new TextEncoder();

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

type TestStatus = "pending" | "running" | "pass" | "fail" | "skip";
interface TestRow {
  path: string;
  status: TestStatus;
  output: string;
  ms?: number;
}

declare global {
  interface Window {
    __numblTestResults?: {
      total: number;
      pass: number;
      fail: number;
      skipped: number;
      failed: { path: string; output: string }[];
      durationMs: number;
    };
  }
}

function runOne(
  testPath: string,
  entry: ManifestEntry,
  sources: Record<string, string>,
  sourceBytes: Map<string, Uint8Array>
): { ok: boolean; output: string } {
  const vfs = new VirtualFileSystem();
  for (const [rel, bytes] of sourceBytes) {
    vfs.writeFile(`${VFS_ROOT}/${rel}`, bytes);
  }
  vfs.clearChangeTracking();

  const system = new BrowserSystemAdapter(vfs);
  const mainFileName = `${VFS_ROOT}/${testPath}`;
  const scriptDir = mainFileName.slice(0, mainFileName.lastIndexOf("/"));
  const workspace: WorkspaceFile[] = entry.workspace.map(rel => ({
    name: `${VFS_ROOT}/${rel}`,
    source: sources[rel],
  }));

  try {
    const result = executeCode(
      sources[testPath],
      {
        displayResults: true,
        fileIO: new BrowserFileIOAdapter(vfs),
        system,
      },
      workspace,
      mainFileName,
      [scriptDir]
    );
    const outputText = result.output.join("");
    const lines = outputText.split("\n").filter(l => l.length > 0);
    const last = lines.length > 0 ? lines[lines.length - 1] : "";
    return { ok: last === "SUCCESS", output: outputText };
  } catch (err) {
    return {
      ok: false,
      output: err instanceof Error ? (err.stack ?? err.message) : String(err),
    };
  }
}

export default function App() {
  const [manifestLoaded, setManifestLoaded] = useState(false);
  const [rows, setRows] = useState<TestRow[]>([]);
  const [done, setDone] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggle = (i: number) =>
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const runAll = async (m: Manifest) => {
    // Install the qhull Delaunay backend before running scripts (mirrors the
    // worker and CLI) so delaunay/delaunayn are available.
    await ensureQhullBackend();
    const t0 = performance.now();
    const updated: TestRow[] = m.tests.map(t => ({
      path: t.path,
      status: "pending",
      output: "",
    }));
    setRows([...updated]);

    // Encode each text source once; decode binary fixtures once. The
    // Uint8Arrays are safe to share across per-test VFS instances (reads
    // don't mutate, fopen snapshots bytes).
    const sourceBytes = new Map<string, Uint8Array>();
    for (const [rel, text] of Object.entries(m.sources)) {
      sourceBytes.set(rel, TEXT_ENCODER.encode(text));
    }
    for (const [rel, b64] of Object.entries(m.binaries)) {
      sourceBytes.set(rel, base64ToBytes(b64));
    }

    for (let i = 0; i < m.tests.length; i++) {
      const entry = m.tests[i];

      if (entry.skip) {
        updated[i] = {
          path: entry.path,
          status: "skip",
          output: `Skipped: ${entry.skip}`,
        };
        setRows([...updated]);
        continue;
      }

      updated[i] = { ...updated[i], status: "running" };
      setRows([...updated]);
      await new Promise(r => setTimeout(r, 0));

      const start = performance.now();
      let row: TestRow;
      try {
        const { ok, output } = runOne(
          entry.path,
          entry,
          m.sources,
          sourceBytes
        );
        row = {
          path: entry.path,
          status: ok ? "pass" : "fail",
          output,
          ms: performance.now() - start,
        };
      } catch (err) {
        row = {
          path: entry.path,
          status: "fail",
          output:
            err instanceof Error ? (err.stack ?? err.message) : String(err),
          ms: performance.now() - start,
        };
      }
      updated[i] = row;
      setRows([...updated]);
    }

    const total = updated.length;
    const pass = updated.filter(r => r.status === "pass").length;
    const fail = updated.filter(r => r.status === "fail").length;
    const skipped = updated.filter(r => r.status === "skip").length;
    const failed = updated
      .filter(r => r.status === "fail")
      .map(r => ({ path: r.path, output: r.output }));
    const durationMs = performance.now() - t0;
    window.__numblTestResults = {
      total,
      pass,
      fail,
      skipped,
      failed,
      durationMs,
    };
    setDone(true);
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch("../test-scripts/manifest.json");
        if (!resp.ok) throw new Error(`manifest fetch → ${resp.status}`);
        const m = (await resp.json()) as Manifest;
        if (cancelled) return;
        setManifestLoaded(true);
        setRows(
          m.tests.map(t => ({ path: t.path, status: "pending", output: "" }))
        );
        await runAll(m);
      } catch (err) {
        if (cancelled) return;
        setRows([
          {
            path: "(manifest)",
            status: "fail",
            output: err instanceof Error ? err.message : String(err),
          },
        ]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const total = rows.length;
  const pass = rows.filter(r => r.status === "pass").length;
  const fail = rows.filter(r => r.status === "fail").length;
  const skip = rows.filter(r => r.status === "skip").length;
  const pending = rows.filter(r => r.status === "pending").length;

  return (
    <div
      style={{
        fontFamily:
          "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        margin: "1rem",
        color: "#111",
      }}
    >
      <h1 style={{ marginTop: 0 }}>numbl browser test runner</h1>
      <p style={{ color: "#555" }}>
        Executes every integration script from <code>numbl_test_scripts/</code>{" "}
        in this browser tab using the shared <code>executeCode</code> entry
        point (same code path as the CLI and unit tests).
      </p>
      <div style={{ marginBottom: "0.75rem" }}>
        <strong>Total:</strong> {total} &nbsp; <strong>Pass:</strong>{" "}
        <span style={{ color: "#0a7" }}>{pass}</span> &nbsp;{" "}
        <strong>Fail:</strong>{" "}
        <span style={{ color: fail > 0 ? "#c33" : "#555" }}>{fail}</span> &nbsp;{" "}
        <strong>Skip:</strong> <span style={{ color: "#888" }}>{skip}</span>{" "}
        &nbsp; <strong>Pending:</strong> {pending} &nbsp;{" "}
        {done ? (
          <em style={{ color: "#0a7" }}>done</em>
        ) : manifestLoaded ? (
          <em>running…</em>
        ) : (
          <em>loading manifest…</em>
        )}
      </div>
      <table
        style={{
          borderCollapse: "collapse",
          width: "100%",
          fontSize: "0.9rem",
        }}
      >
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "1px solid #ddd" }}>
            <th style={{ padding: "4px 4px", width: "1.5em" }}></th>
            <th style={{ padding: "4px 8px" }}>#</th>
            <th style={{ padding: "4px 8px" }}>Status</th>
            <th style={{ padding: "4px 8px" }}>Script</th>
            <th style={{ padding: "4px 8px" }}>ms</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const isOpen = expanded.has(i);
            const hasOutput = r.output.length > 0;
            return (
              <Fragment key={r.path}>
                <tr
                  style={{
                    borderBottom: isOpen ? "none" : "1px solid #f0f0f0",
                    verticalAlign: "top",
                  }}
                >
                  <td style={{ padding: "2px 4px", textAlign: "center" }}>
                    {hasOutput ? (
                      <button
                        onClick={() => toggle(i)}
                        title={isOpen ? "collapse" : "expand"}
                        style={{
                          border: "none",
                          background: "transparent",
                          cursor: "pointer",
                          fontFamily: "monospace",
                          fontSize: "0.9rem",
                          padding: 0,
                          color: "#666",
                        }}
                      >
                        {isOpen ? "▼" : "▶"}
                      </button>
                    ) : null}
                  </td>
                  <td style={{ padding: "2px 8px", color: "#888" }}>{i + 1}</td>
                  <td
                    style={{
                      padding: "2px 8px",
                      color:
                        r.status === "pass"
                          ? "#0a7"
                          : r.status === "fail"
                            ? "#c33"
                            : r.status === "running"
                              ? "#06b"
                              : r.status === "skip"
                                ? "#b77"
                                : "#888",
                      fontWeight: r.status === "fail" ? 600 : 400,
                    }}
                  >
                    {r.status.toUpperCase()}
                  </td>
                  <td style={{ padding: "2px 8px", fontFamily: "monospace" }}>
                    <a
                      href={`${GITHUB_BASE}/${r.path}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: "#06b", textDecoration: "none" }}
                    >
                      {r.path}
                    </a>
                  </td>
                  <td style={{ padding: "2px 8px", color: "#888" }}>
                    {r.ms != null ? r.ms.toFixed(0) : ""}
                  </td>
                </tr>
                {isOpen && hasOutput && (
                  <tr
                    key={r.path + "::out"}
                    style={{ borderBottom: "1px solid #f0f0f0" }}
                  >
                    <td colSpan={5} style={{ padding: "0 8px 8px 24px" }}>
                      <pre
                        style={{
                          margin: 0,
                          padding: "6px 8px",
                          background:
                            r.status === "fail" ? "#fff4f4" : "#f7f7f7",
                          border:
                            r.status === "fail"
                              ? "1px solid #f1c9c9"
                              : "1px solid #e4e4e4",
                          whiteSpace: "pre-wrap",
                          fontSize: "0.8rem",
                          maxHeight: "30em",
                          overflow: "auto",
                        }}
                      >
                        {r.output}
                      </pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
