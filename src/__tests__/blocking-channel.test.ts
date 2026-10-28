/// <reference types="node" />

/**
 * The blocking channel (numbl-core/runtime/blockingChannel.ts) between this
 * thread (the blocked client, like the interpreter worker) and a
 * worker_threads server running the same server source a browser helper
 * worker gets: round trips, typed-array payloads that grow the region,
 * fire-and-forget posts, server errors, and cancellation mid-wait.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { once } from "node:events";
import { Worker } from "node:worker_threads";
import {
  BlockingChannel,
  channelServerSource,
  clearCancel,
  createCancelBuffer,
} from "../numbl-core/runtime/blockingChannel.js";
import { CancellationError } from "../numbl-core/runtime/error.js";

const SERVER = `
const { parentPort, workerData } = require("node:worker_threads");
const createChannelServer = ${channelServerSource()};
const posts = [];
let cancels = 0;
// Plays the host: sets the cancel flag, optionally waking the client.
function cancelLater(ms, notify) {
  setTimeout(() => {
    const w = new Int32Array(workerData);
    Atomics.store(w, 0, 1);
    if (notify) Atomics.notify(w, 1);
  }, ms);
}
const server = createChannelServer({
  request(cmd, body, reply) {
    if (cmd === "echo") reply.ok(body);
    else if (cmd === "ramp") {
      const x = new Float64Array(body.n);
      for (let i = 0; i < x.length; i++) x[i] = i;
      reply.ok({ x });
    } else if (cmd === "fail") reply.error("boom");
    else if (cmd === "throw") throw new Error("thrown");
    else if (cmd === "later") setTimeout(() => reply.ok(body.value), body.ms);
    else if (cmd === "hang") {
      cancelLater(body.cancelMs, body.notify);
      if (body.replyMs) setTimeout(() => reply.ok("stale"), body.replyMs);
    } else if (cmd === "stats") reply.ok({ posts, cancels });
    else reply.error("unknown " + cmd);
  },
  post(cmd, body) {
    if (cmd === "cancelLater") cancelLater(body.ms, true);
    else posts.push([cmd, body]);
  },
  cancel() {
    cancels++;
  },
});
parentPort.on("message", m => server.handle(m));
`;

let worker: Worker;
let cancel: SharedArrayBuffer;
let channel: BlockingChannel;

beforeAll(async () => {
  cancel = createCancelBuffer()!;
  worker = new Worker(SERVER, { eval: true, workerData: cancel });
  await once(worker, "online");
  // A long slice, so a prompt wake can only come from the notify.
  channel = new BlockingChannel(cancel, 5000);
});

afterAll(async () => {
  await worker.terminate();
});

function timed<T>(fn: () => T): { value?: T; error?: unknown; ms: number } {
  const t0 = performance.now();
  try {
    return { value: fn(), ms: performance.now() - t0 };
  } catch (error) {
    return { error, ms: performance.now() - t0 };
  }
}

describe("BlockingChannel", () => {
  it("round-trips JSON-like values with typed arrays and non-finite numbers", () => {
    const body = {
      s: "héllo",
      n: [1, -2.5, NaN, Infinity, -Infinity],
      flag: true,
      none: null,
      f32: new Float32Array([1.5, 2.5]),
      i32: new Int32Array([-1, 7]),
      u8: new Uint8Array([3, 4, 5]),
      nested: { f64: new Float64Array([Math.PI]) },
    };
    const r = channel.request(worker, "echo", body) as typeof body;
    expect(r.s).toBe("héllo");
    expect(r.n).toEqual([1, -2.5, NaN, Infinity, -Infinity]);
    expect(r.flag).toBe(true);
    expect(r.none).toBe(null);
    expect(r.f32).toBeInstanceOf(Float32Array);
    expect(Array.from(r.f32)).toEqual([1.5, 2.5]);
    expect(r.i32).toBeInstanceOf(Int32Array);
    expect(Array.from(r.i32)).toEqual([-1, 7]);
    expect(Array.from(r.u8)).toEqual([3, 4, 5]);
    expect(r.nested.f64[0]).toBe(Math.PI);
    // The reply does not alias the shared region.
    expect(r.f32.buffer).not.toBeInstanceOf(SharedArrayBuffer);
    expect(channel.request(worker, "echo", "plain")).toBe("plain");
  });

  it("grows the region for large replies and requests", () => {
    const n = 1 << 20; // 8 MB of doubles
    const big = channel.request(worker, "ramp", { n }) as { x: Float64Array };
    expect(big.x.length).toBe(n);
    expect(big.x[n - 1]).toBe(n - 1);
    const x = new Float64Array(2 * n).map((_, i) => i * 0.5);
    const echoed = channel.request(worker, "echo", { x }) as {
      x: Float64Array;
    };
    expect(echoed.x.length).toBe(x.length);
    let same = true;
    for (let i = 0; i < x.length; i++) same &&= echoed.x[i] === x[i];
    expect(same).toBe(true);
  });

  it("delivers posts in order before a later request", () => {
    channel.post(worker, "note", { a: 1 });
    channel.post(worker, "note", "two");
    const stats = channel.request(worker, "stats") as {
      posts: unknown[];
    };
    expect(stats.posts).toEqual([
      ["note", { a: 1 }],
      ["note", "two"],
    ]);
  });

  it("throws server errors", () => {
    expect(() => channel.request(worker, "fail")).toThrow("boom");
    expect(() => channel.request(worker, "throw")).toThrow("thrown");
  });

  it("waits for a slow server", () => {
    const r = timed(() =>
      channel.request(worker, "later", { ms: 150, value: 42 })
    );
    expect(r.value).toBe(42);
    expect(r.ms).toBeGreaterThanOrEqual(140);
  });

  it("is cancelled promptly mid-wait, and a late reply is dropped", () => {
    const r = timed(() =>
      channel.request(worker, "hang", {
        cancelMs: 50,
        notify: true,
        replyMs: 150,
      })
    );
    expect(r.error).toBeInstanceOf(CancellationError);
    // Woken by the notify, well before the 5 s slice ends.
    expect(r.ms).toBeLessThan(1000);
    clearCancel(cancel);
    // The stale "hang" reply arrives while this request is pending.
    const next = channel.request(worker, "later", { ms: 300, value: "fresh" });
    expect(next).toBe("fresh");
    const stats = channel.request(worker, "stats") as { cancels: number };
    expect(stats.cancels).toBe(1);
  });

  it("notices a cancel without a notify at the next slice", () => {
    const sliced = new BlockingChannel(cancel, 20);
    const r = timed(() =>
      sliced.request(worker, "hang", { cancelMs: 50, notify: false })
    );
    expect(r.error).toBeInstanceOf(CancellationError);
    expect(r.ms).toBeLessThan(1000);
    clearCancel(cancel);
  });

  it("sleeps, and a cancel wakes the sleep", () => {
    const r = timed(() => channel.sleep(100));
    expect(r.error).toBeUndefined();
    expect(r.ms).toBeGreaterThanOrEqual(95);

    // pause(60) in the IDE: Stop ends it at once.
    channel.post(worker, "cancelLater", { ms: 50 });
    const stopped = timed(() => channel.sleep(60_000));
    expect(stopped.error).toBeInstanceOf(CancellationError);
    expect(stopped.ms).toBeLessThan(1000);
    clearCancel(cancel);
  });
});
