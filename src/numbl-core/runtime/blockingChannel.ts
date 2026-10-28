/**
 * The one way the synchronous interpreter thread waits on another thread:
 * `input()`, `pause`, and `gpu.call` all block here, so Stop interrupts
 * every one of them. SharedArrayBuffer + Atomics only; no DOM or Node APIs.
 *
 * Roles. The *client* is the interpreter thread, which blocks. A *server* is
 * a thread with an event loop (the page's main thread, a helper worker, a
 * Node worker_threads worker) that receives the channel's messages through
 * `postMessage` and answers through shared memory.
 *
 * Shared memory.
 *   ctrl  The host's cancel buffer (`createCancelBuffer`, 8 bytes):
 *           word 0  cancel flag, the one the interpreter and JIT poll;
 *           word 1  state, `seq << 2 | status` (PENDING, OK, ERR, GROW).
 *         A 4-byte cancel buffer still works; the state word then lives in a
 *         private buffer and Stop is noticed at the next slice, not at once.
 *   data  The reply region, owned by the client and kept across requests.
 *         Word 0 holds the byte length of the reply frame at offset 8. A
 *         frame is JSON plus the raw bytes of any typed arrays, so a typed
 *         array is copied once into the region and once out of it.
 *
 * Request. The client stores PENDING with a fresh `seq`, posts
 * `{ type: "blocking_channel", op: "req" }` with the body, and waits on the
 * state word in slices of `sliceMs`, checking the cancel flag after each.
 * `signalCancel` also notifies the state word, so Stop wakes it at once. On
 * cancel the client posts `op: "cancel"`, abandons the region (a late reply
 * then writes into memory nobody reads) and throws CancellationError. The
 * server writes the reply into the region and moves the state from PENDING
 * to OK or ERR with a compare-exchange on the request's `seq`, so a stale
 * reply never completes a later request. A reply that does not fit moves the
 * state to GROW with the size in word 0; the client allocates a larger region
 * and posts `op: "fetch"` for the server to write into.
 *
 * Bodies travel in the message (structured clone): a blocked client cannot
 * receive messages, so only replies need shared memory, and V8 copies into a
 * SharedArrayBuffer at about half the speed of a plain copy. `post` is
 * fire-and-forget. `sleep` waits on the state word with nobody to answer,
 * which makes `pause` stoppable.
 */

import { CancellationError } from "./error.js";

/** Anything messages can be posted to: a Worker, a MessagePort, a worker's
 *  global scope, a Node worker_threads Worker or port. */
export interface ChannelTarget {
  postMessage(msg: unknown): void;
}

/** The messages a server receives; route them to `ChannelServer.handle`. */
export interface ChannelMessage {
  type: "blocking_channel";
  op: "req" | "post" | "fetch" | "cancel";
  seq: number;
  cmd?: string;
  body?: unknown;
  ctrl?: SharedArrayBuffer;
  slot?: number;
  data?: SharedArrayBuffer;
}

/** How a server answers one request. Only the first answer counts. The
 *  value may be JSON-like data (non-finite numbers included) with typed
 *  arrays anywhere inside. */
export interface ChannelReply {
  ok(value: unknown): void;
  error(message: string): void;
}

export interface ChannelServerHandlers {
  request(cmd: string, body: unknown, reply: ChannelReply): void;
  post?(cmd: string, body: unknown): void;
  /** The client stopped waiting for the current request (Stop). */
  cancel?(): void;
}

export interface ChannelServer {
  /** Handle `msg` if it is a channel message; returns whether it was. */
  handle(msg: unknown): boolean;
}

const P = { PENDING: 0, OK: 1, ERR: 2, GROW: 3, HEADER: 8 };

const MIN_REGION_BYTES = 1 << 16;

// ── Reply frames ────────────────────────────────────────────────────────
// Self-contained (no outer references), so channelServerSource() can embed
// them in a helper worker's source with toString().

interface Frame {
  size: number;
  write(dst: Uint8Array): void;
}

function encodeFrame(value: unknown): Frame {
  const chunks: { v: ArrayBufferView; off: number }[] = [];
  let bin = 0;
  const json = JSON.stringify(value, function (_k, v) {
    if (typeof v === "number" && !isFinite(v)) return { __n: String(v) };
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
      const off = bin;
      chunks.push({ v, off });
      bin = (bin + v.byteLength + 7) & ~7;
      return {
        __ta: v.constructor.name,
        o: off,
        n: (v as unknown as { length: number }).length,
      };
    }
    return v;
  });
  const text = new TextEncoder().encode(json === undefined ? "null" : json);
  const head = (4 + text.length + 7) & ~7;
  return {
    size: head + bin,
    write(dst) {
      new DataView(dst.buffer, dst.byteOffset, 4).setUint32(
        0,
        text.length,
        true
      );
      dst.set(text, 4);
      for (const c of chunks) {
        dst.set(
          new Uint8Array(c.v.buffer, c.v.byteOffset, c.v.byteLength),
          head + c.off
        );
      }
    },
  };
}

/** Decode a frame. Typed arrays are copied out, so the result does not alias
 *  `src` (which the next reply overwrites). */
function decodeFrame(src: Uint8Array): unknown {
  const len = new DataView(src.buffer, src.byteOffset, 4).getUint32(0, true);
  // TextDecoder rejects views on a SharedArrayBuffer: copy the JSON first.
  const json = new TextDecoder().decode(src.slice(4, 4 + len));
  const head = (4 + len + 7) & ~7;
  const names = [
    "Float64Array",
    "Float32Array",
    "Int32Array",
    "Uint32Array",
    "Int16Array",
    "Uint16Array",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
  ];
  return JSON.parse(json, function (_k, v) {
    if (v && typeof v === "object") {
      if (typeof v.__n === "string") return Number(v.__n);
      if (typeof v.__ta === "string") {
        if (names.indexOf(v.__ta) < 0)
          throw new Error("unsupported typed array " + v.__ta);
        const C = (
          globalThis as unknown as Record<string, Int8ArrayConstructor>
        )[v.__ta];
        return new C(new C(src.buffer, src.byteOffset + head + v.o, v.n));
      }
    }
    return v;
  });
}

// ── Server ──────────────────────────────────────────────────────────────

function channelServerFactory(p: typeof P, encode: typeof encodeFrame) {
  return function createChannelServer(
    handlers: ChannelServerHandlers
  ): ChannelServer {
    let current: ChannelMessage | null = null;
    let grow: { seq: number; status: number; frame: Frame } | null = null;

    // Write `frame` into `data` and complete request `m`, unless the client
    // has moved on (cancelled, or started another request).
    function complete(
      m: ChannelMessage,
      data: SharedArrayBuffer,
      status: number,
      frame: Frame
    ) {
      const st = new Int32Array(m.ctrl!);
      const want = m.seq * 4 + p.PENDING;
      if (Atomics.load(st, m.slot!) !== want) return;
      let next = status;
      if (p.HEADER + frame.size > data.byteLength) {
        grow = { seq: m.seq, status, frame };
        next = p.GROW;
      } else {
        frame.write(new Uint8Array(data, p.HEADER, frame.size));
      }
      Atomics.store(new Int32Array(data, 0, 1), 0, frame.size);
      if (Atomics.compareExchange(st, m.slot!, want, m.seq * 4 + next) === want)
        Atomics.notify(st, m.slot!);
    }

    return {
      handle(msg: unknown): boolean {
        const m = msg as ChannelMessage;
        if (!m || typeof m !== "object" || m.type !== "blocking_channel")
          return false;
        if (m.op === "post") {
          handlers.post?.(m.cmd!, m.body);
        } else if (m.op === "req") {
          current = m;
          grow = null;
          let done = false;
          const finish = (status: number, value: unknown) => {
            if (done) return;
            done = true;
            let frame: Frame;
            try {
              frame = encode(value);
            } catch (err) {
              status = p.ERR;
              frame = encode(String(err));
            }
            complete(m, m.data!, status, frame);
          };
          try {
            handlers.request(m.cmd!, m.body, {
              ok: v => finish(p.OK, v),
              error: e => finish(p.ERR, String(e)),
            });
          } catch (err) {
            finish(p.ERR, String(err));
          }
        } else if (m.op === "fetch") {
          if (grow && current && grow.seq === m.seq) {
            const g = grow;
            grow = null;
            complete(current, m.data!, g.status, g.frame);
          }
        } else if (m.op === "cancel") {
          if (current && current.seq === m.seq) {
            current = null;
            grow = null;
            handlers.cancel?.();
          }
        }
        return true;
      },
    };
  };
}

/** Make a server for channel messages arriving on this thread. */
export const createChannelServer = channelServerFactory(P, encodeFrame);

/** JavaScript source of an expression that evaluates to
 *  `createChannelServer`, for helper workers started from a Blob. */
export function channelServerSource(): string {
  // `__name`: the helper esbuild's keepNames (tsx, some bundlers) inserts.
  return `(() => { const __name = f => f; return (${channelServerFactory.toString()})(${JSON.stringify(P)}, ${encodeFrame.toString()}); })()`;
}

// ── Client ──────────────────────────────────────────────────────────────

/** An 8-byte cancel buffer (word 0: flag, word 1: channel state), or null
 *  without SharedArrayBuffer (no cross-origin isolation). */
export function createCancelBuffer(): SharedArrayBuffer | null {
  return typeof SharedArrayBuffer === "undefined"
    ? null
    : new SharedArrayBuffer(8);
}

/** Host side of Stop: set the cancel flag and wake a blocked channel. */
export function signalCancel(cancel: SharedArrayBuffer): void {
  const w = new Int32Array(cancel);
  Atomics.store(w, 0, 1);
  if (w.length > 1) Atomics.notify(w, 1);
}

/** Clear the cancel flag before the next run. */
export function clearCancel(cancel: SharedArrayBuffer): void {
  Atomics.store(new Int32Array(cancel), 0, 0);
}

export class BlockingChannel {
  private readonly cancelFlag: Int32Array | null;
  private readonly ctrl: SharedArrayBuffer;
  private readonly slot: number;
  private readonly state: Int32Array;
  private data: SharedArrayBuffer | null = null;
  private seq = 0;

  /**
   * @param cancel The host's cancel buffer, or null for a channel that
   *   cannot be cancelled.
   * @param sliceMs Longest single Atomics.wait between cancel checks.
   */
  constructor(
    cancel: SharedArrayBuffer | null,
    private readonly sliceMs = 50
  ) {
    this.cancelFlag = cancel ? new Int32Array(cancel, 0, 1) : null;
    if (cancel && cancel.byteLength >= 8) {
      this.ctrl = cancel;
      this.slot = 1;
    } else {
      this.ctrl = new SharedArrayBuffer(4);
      this.slot = 0;
    }
    this.state = new Int32Array(this.ctrl);
  }

  /** Send `cmd` with `body` (structured-cloned) to `target` and block until
   *  it answers. Returns the reply; a server error is thrown as an Error.
   *  Throws CancellationError if the cancel flag is set while waiting. */
  request(target: ChannelTarget, cmd: string, body?: unknown): unknown {
    const seq = this.nextSeq();
    const pending = seq * 4 + P.PENDING;
    Atomics.store(this.state, this.slot, pending);
    this.send(target, {
      op: "req",
      seq,
      cmd,
      body,
      ctrl: this.ctrl,
      slot: this.slot,
      data: this.region(MIN_REGION_BYTES),
    });
    for (;;) {
      const word = this.wait(pending, Infinity, target, seq);
      const data = this.data!;
      const len = Atomics.load(new Int32Array(data, 0, 1), 0);
      if ((word & 3) === P.GROW) {
        // Reset before asking, so the server's compare-exchange can succeed.
        Atomics.store(this.state, this.slot, pending);
        this.send(target, {
          op: "fetch",
          seq,
          data: this.region(P.HEADER + len),
        });
        continue;
      }
      const value = decodeFrame(new Uint8Array(data, P.HEADER, len));
      if ((word & 3) === P.ERR) throw new Error(String(value));
      return value;
    }
  }

  /** Send `cmd` with `body` (structured-cloned) without waiting. */
  post(target: ChannelTarget, cmd: string, body?: unknown): void {
    this.send(target, { op: "post", seq: 0, cmd, body });
  }

  /** Block for `ms` milliseconds; throws CancellationError on cancel. */
  sleep(ms: number): void {
    if (!(ms > 0)) return;
    const pending = this.nextSeq() * 4 + P.PENDING;
    Atomics.store(this.state, this.slot, pending);
    this.wait(pending, ms, null, 0);
  }

  private send(target: ChannelTarget, msg: Omit<ChannelMessage, "type">): void {
    target.postMessage({ type: "blocking_channel", ...msg });
  }

  private nextSeq(): number {
    this.seq = (this.seq + 1) & 0x1fffffff;
    return this.seq;
  }

  /** The reply region, at least `bytes` long. */
  private region(bytes: number): SharedArrayBuffer {
    if (this.data && this.data.byteLength >= bytes) return this.data;
    let size = Math.max(MIN_REGION_BYTES, this.data?.byteLength ?? 0);
    while (size < bytes) size *= 2;
    this.data = new SharedArrayBuffer(size);
    return this.data;
  }

  /** Wait while the state word is `pending`, up to `ms`. Returns the state
   *  word (still `pending` on timeout). */
  private wait(
    pending: number,
    ms: number,
    target: ChannelTarget | null,
    seq: number
  ): number {
    const deadline = Date.now() + ms;
    for (;;) {
      const word = Atomics.load(this.state, this.slot);
      if (word !== pending) return word;
      if (this.cancelFlag && Atomics.load(this.cancelFlag, 0) !== 0) {
        if (target) {
          // Abandon the region so a late reply cannot touch the next request.
          this.data = null;
          this.send(target, { op: "cancel", seq });
        }
        throw new CancellationError();
      }
      const left = deadline - Date.now();
      if (left <= 0) return word;
      Atomics.wait(
        this.state,
        this.slot,
        pending,
        Math.min(left, this.sliceMs)
      );
    }
  }
}

const channels = new WeakMap<SharedArrayBuffer, BlockingChannel>();

/** The channel bound to a cancel buffer (one per buffer, so its reply region
 *  is reused across runs). */
export function blockingChannelFor(cancel: SharedArrayBuffer): BlockingChannel {
  let ch = channels.get(cancel);
  if (!ch) {
    ch = new BlockingChannel(cancel);
    channels.set(cancel, ch);
  }
  return ch;
}

let sleepWarned = false;

/** Block for `ms` milliseconds, through `channel` when there is one (so the
 *  wait is cancellable). Busy-waits without SharedArrayBuffer. */
export function sleepBlocking(
  ms: number,
  channel: BlockingChannel | null
): void {
  if (!(ms > 0)) return;
  if (channel) return channel.sleep(ms);
  if (typeof SharedArrayBuffer !== "undefined") {
    return new BlockingChannel(null).sleep(ms);
  }
  if (!sleepWarned) {
    sleepWarned = true;
    console.warn(
      "SharedArrayBuffer is not available, so pause() will busy-wait. " +
        "Enable Cross-Origin Isolation headers (COOP/COEP) for efficient blocking."
    );
  }
  const end = Date.now() + ms;
  while (Date.now() < end) {
    /* spin */
  }
}
