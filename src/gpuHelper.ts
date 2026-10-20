/**
 * Synchronous WebGPU access for .numbl.js functions (browser worker only).
 *
 * Scripts run synchronously in the numbl worker, but WebGPU is async. A helper
 * worker owns the GPU device; the numbl worker posts a request and blocks on a
 * SharedArrayBuffer with Atomics.wait until the helper writes the result.
 *
 * The helper must be spawned at worker boot: browsers do not start a worker
 * until its creating thread returns to the event loop, so spawning it from
 * inside a run (which never yields) would deadlock.
 *
 * .numbl.js usage (the injected `gpu` global is null when unavailable):
 *   gpu.load("name", function () { return { method: async (device, args) => result }; });
 *   var result = gpu.call("name", "method", args);
 * The factory runs inside the helper, so it must be self-contained; its state
 * (e.g. GPU buffers) persists across calls and runs until the source changes,
 * when an optional dispose() method is called. Arguments
 * and results may be numbers, strings, booleans, null, arrays, plain objects
 * and typed arrays.
 */

export interface GpuHelper {
  load(name: string, factory: (() => unknown) | string): void;
  call(name: string, method: string, args?: unknown): unknown;
}

// ctrl[0] status values
const PENDING = 0;
const OK = 1;
const ERR = 2;
const NEED_BUF = 3;

const INITIAL_BUF_BYTES = 1 << 20;
const CALL_TIMEOUT_MS = 10 * 60 * 1000;

type TypedArrayName =
  | "Float64Array"
  | "Float32Array"
  | "Int32Array"
  | "Uint32Array"
  | "Uint8Array";

// Shared by both sides; serialized into the helper source via toString().
function encodeValue(value: unknown): Uint8Array {
  const chunks: { arr: ArrayBufferView; off: number }[] = [];
  let binLen = 0;
  const json = JSON.stringify(value, function (_k, v) {
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
      const off = binLen;
      chunks.push({ arr: v, off });
      binLen += v.byteLength;
      binLen = (binLen + 7) & ~7;
      return {
        __ta: v.constructor.name,
        o: off,
        n: (v as unknown as { length: number }).length,
      };
    }
    return v;
  });
  const jsonBytes = new TextEncoder().encode(
    json === undefined ? "null" : json
  );
  const head = (4 + jsonBytes.length + 7) & ~7;
  const out = new Uint8Array(head + binLen);
  new DataView(out.buffer).setUint32(0, jsonBytes.length, true);
  out.set(jsonBytes, 4);
  for (const c of chunks) {
    out.set(
      new Uint8Array(c.arr.buffer, c.arr.byteOffset, c.arr.byteLength),
      head + c.off
    );
  }
  return out;
}

function decodeValue(bytes: Uint8Array): unknown {
  const jsonLen = new DataView(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength
  ).getUint32(0, true);
  // TextDecoder rejects views on a SharedArrayBuffer, so copy first.
  const json = new TextDecoder().decode(bytes.slice(4, 4 + jsonLen));
  const head = (4 + jsonLen + 7) & ~7;
  const ctors: Record<TypedArrayName, { BYTES_PER_ELEMENT: number }> = {
    Float64Array,
    Float32Array,
    Int32Array,
    Uint32Array,
    Uint8Array,
  };
  return JSON.parse(json, function (_k, v) {
    if (v && typeof v === "object" && typeof v.__ta === "string") {
      const C = ctors[v.__ta as TypedArrayName] as unknown as {
        new (b: ArrayBuffer): ArrayBufferView;
        BYTES_PER_ELEMENT: number;
      };
      if (!C) throw new Error("gpu: unsupported typed array " + v.__ta);
      const start = bytes.byteOffset + head + v.o;
      const copy = bytes.buffer.slice(
        start,
        start + v.n * C.BYTES_PER_ELEMENT
      ) as ArrayBuffer;
      return new C(copy);
    }
    return v;
  });
}

function helperMain(
  encode: (v: unknown) => Uint8Array,
  status: { OK: number; ERR: number; NEED_BUF: number }
) {
  const modules = new Map<
    string,
    { source: string; mod?: Record<string, unknown>; error?: string }
  >();
  const pending = new Map<number, { st: number; bytes: Uint8Array }>();
  let devP: Promise<GPUDevice> | null = null;
  function device(): Promise<GPUDevice> {
    if (!devP) {
      devP = navigator.gpu.requestAdapter().then(a => {
        if (!a) throw new Error("gpu: no WebGPU adapter");
        return a.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
            maxBufferSize: a.limits.maxBufferSize,
          },
        });
      });
    }
    return devP;
  }
  function deliver(
    ctrl: Int32Array,
    buf: SharedArrayBuffer,
    st: number,
    bytes: Uint8Array
  ) {
    Atomics.store(ctrl, 1, bytes.byteLength);
    if (bytes.byteLength <= buf.byteLength) {
      new Uint8Array(buf).set(bytes);
      Atomics.store(ctrl, 0, st);
    } else {
      Atomics.store(ctrl, 0, status.NEED_BUF);
    }
    Atomics.notify(ctrl, 0);
  }
  self.onmessage = async (e: MessageEvent) => {
    const m = e.data;
    if (m.type === "init") {
      const ready = new Int32Array(m.ready);
      Atomics.store(ready, 0, 1);
      Atomics.notify(ready, 0);
    } else if (m.type === "load") {
      const prev = modules.get(m.name);
      if (prev && prev.source === m.source) return;
      // A module that keeps GPU resources across calls releases them in
      // dispose() when it is replaced by a new version.
      const dispose = prev?.mod?.dispose;
      if (typeof dispose === "function") {
        try {
          dispose();
        } catch {
          // ignore
        }
      }
      try {
        const mod = new Function("return (" + m.source + ")();")();
        modules.set(m.name, { source: m.source, mod });
      } catch (err) {
        modules.set(m.name, { source: m.source, error: String(err) });
      }
    } else if (m.type === "call") {
      const ctrl = new Int32Array(m.ctrl);
      let st = status.OK;
      let bytes: Uint8Array;
      try {
        const entry = modules.get(m.name);
        if (!entry) throw new Error("gpu: module '" + m.name + "' not loaded");
        if (entry.error) throw new Error(entry.error);
        const fn = entry.mod![m.method];
        if (typeof fn !== "function")
          throw new Error("gpu: '" + m.name + "' has no method " + m.method);
        const result = await fn(await device(), m.args);
        bytes = encode(result);
      } catch (err) {
        st = status.ERR;
        bytes = encode(
          String((err as Error)?.stack || (err as Error)?.message || err)
        );
      }
      pending.set(m.id, { st, bytes });
      deliver(ctrl, m.buf, st, bytes);
    } else if (m.type === "fetch") {
      const p = pending.get(m.id)!;
      pending.delete(m.id);
      deliver(new Int32Array(m.ctrl), m.buf, p.st, p.bytes);
    }
  };
}

function helperSource(): string {
  return (
    "const encodeValue = " +
    encodeValue.toString() +
    ";\n(" +
    helperMain.toString() +
    ")(encodeValue, " +
    JSON.stringify({ OK, ERR, NEED_BUF }) +
    ");\n"
  );
}

/**
 * Spawn the GPU helper. Call at worker boot, before any run. Returns null when
 * WebGPU or SharedArrayBuffer (cross-origin isolation) is unavailable.
 */
export function startGpuHelper(): GpuHelper | null {
  if (
    typeof navigator === "undefined" ||
    !(navigator as Navigator & { gpu?: unknown }).gpu ||
    typeof SharedArrayBuffer === "undefined" ||
    typeof Worker === "undefined"
  ) {
    return null;
  }
  const url = URL.createObjectURL(
    new Blob([helperSource()], { type: "text/javascript" })
  );
  const worker = new Worker(url);
  const ready = new Int32Array(new SharedArrayBuffer(4));
  worker.postMessage({ type: "init", ready: ready.buffer });

  const ctrl = new Int32Array(new SharedArrayBuffer(8));
  let buf = new SharedArrayBuffer(INITIAL_BUF_BYTES);
  let nextId = 1;

  function waitStatus(): number {
    const r = Atomics.wait(ctrl, 0, PENDING, CALL_TIMEOUT_MS);
    if (r === "timed-out") throw new Error("gpu: call timed out");
    return Atomics.load(ctrl, 0);
  }

  return {
    load(name, factory) {
      const source =
        typeof factory === "function" ? factory.toString() : factory;
      worker.postMessage({ type: "load", name, source });
    },
    call(name, method, args) {
      if (Atomics.load(ready, 0) !== 1) {
        throw new Error(
          "gpu: helper worker has not started yet; try running again"
        );
      }
      const id = nextId++;
      Atomics.store(ctrl, 0, PENDING);
      worker.postMessage({
        type: "call",
        id,
        name,
        method,
        args,
        ctrl: ctrl.buffer,
        buf,
      });
      let st = waitStatus();
      if (st === NEED_BUF) {
        const len = Atomics.load(ctrl, 1);
        buf = new SharedArrayBuffer(len);
        Atomics.store(ctrl, 0, PENDING);
        worker.postMessage({ type: "fetch", id, ctrl: ctrl.buffer, buf });
        st = waitStatus();
      }
      const len = Atomics.load(ctrl, 1);
      const value = decodeValue(new Uint8Array(buf, 0, len));
      if (st === ERR) throw new Error(String(value));
      return value;
    },
  };
}
