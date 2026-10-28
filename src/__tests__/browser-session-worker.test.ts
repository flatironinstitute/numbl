/**
 * The NumblSession worker (src/browser/worker.ts) carries workspace state from
 * the boot run into later `execute` calls, and leaves it unchanged when one
 * fails.
 */

import { beforeAll, describe, expect, it } from "vitest";

type WorkerMessage = Record<string, unknown>;

let onmessage: (e: { data: WorkerMessage }) => void;
let posted: WorkerMessage[] = [];

async function waitFor(type: string): Promise<WorkerMessage> {
  for (let i = 0; i < 200; i++) {
    const m = posted.find(p => p.type === type);
    if (m) return m;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error(`no ${type} message`);
}

function execute(code: string): { ok: boolean; output: string } {
  posted = [];
  onmessage({ data: { type: "execute", id: 1, code } });
  const msg = posted.find(m => m.type === "executeResult");
  return msg!.result as { ok: boolean; output: string };
}

beforeAll(async () => {
  (globalThis as unknown as { self: unknown }).self = {
    postMessage: (m: WorkerMessage) => posted.push(m),
    onmessage: null,
  };
  await import("../browser/worker.js");
  onmessage = (
    globalThis as unknown as { self: { onmessage: typeof onmessage } }
  ).self.onmessage;
  posted = [];
  onmessage({
    data: {
      type: "boot",
      files: [{ path: "main.m", content: "a = 5; figure;" }],
      mainFile: "main.m",
      mip: false,
      persistSystem: false,
      systemInactivityMs: 0,
      optimization: "1",
      maxIterations: 1e9,
      displayResults: true,
    },
  });
  await waitFor("ready");
});

describe("browser session worker", () => {
  it("continues from the boot run's workspace and figures", () => {
    const r = execute("h = figure; disp([a + 1, h])");
    expect(r.ok).toBe(true);
    expect(r.output).toBe("   6   2\n");
  });

  it("keeps the workspace after a failed execute, with what it did before the error", () => {
    expect(execute("b = 7; disp(a)").output).toBe("5\n");
    expect(execute("a = 100; error('boom')").ok).toBe(false);
    expect(execute("disp(a); disp(b)").output).toBe("100\n7\n");
  });
});
