/**
 * Request frames for `numbl plot-server`'s `POST /eval`: MATLAB code plus
 * variables to put into the workspace before running it. Two encodings,
 * told apart by the first four bytes.
 *
 * JSON: `{ "code": "...", "vars": { name: value } }`.
 *
 * Binary (all integers little-endian):
 *
 *   bytes 0-3   "NBF1"
 *   bytes 4-7   uint32 header length H
 *   next H      UTF-8 JSON header, same shape as the JSON frame
 *   rest        payload
 *
 * A variable value is one of:
 *   - a number, boolean or string (char row vector)
 *   - `{ shape: [m, n, ...], data: [...] }`: column-major, `null` for NaN
 *   - `{ shape: [m, n, ...], dtype: "float64" | "float32", offset: k }`:
 *     prod(shape) little-endian values at byte `k` of the payload,
 *     column-major. Only allowed in a binary frame.
 */

import type { RuntimeValue } from "./numbl-core/runtime/types.js";
import { RTV } from "./numbl-core/runtime/constructors.js";

export interface PlotFrame {
  code: string;
  vars: Record<string, RuntimeValue>;
}

const MAGIC = [0x4e, 0x42, 0x46, 0x31]; // "NBF1"

const ITEM_SIZE = { float64: 8, float32: 4 } as const;
type Dtype = keyof typeof ITEM_SIZE;

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
  throw new Error("plotFrame: big-endian hosts are not supported");
}

export function isBinaryFrame(body: Uint8Array): boolean {
  return body.length >= 8 && MAGIC.every((b, i) => body[i] === b);
}

/** Decode a frame in either encoding. Throws on malformed input. */
export function decodeFrame(body: Uint8Array): PlotFrame {
  if (isBinaryFrame(body)) {
    const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
    const headerLen = view.getUint32(4, true);
    if (8 + headerLen > body.length) {
      throw new Error("header length exceeds frame size");
    }
    const header = parseJson(body.subarray(8, 8 + headerLen));
    return decodeMessage(header, body.subarray(8 + headerLen));
  }
  return decodeMessage(parseJson(body), null);
}

function parseJson(bytes: Uint8Array): unknown {
  return JSON.parse(new TextDecoder().decode(bytes));
}

function decodeMessage(msg: unknown, payload: Uint8Array | null): PlotFrame {
  if (typeof msg !== "object" || msg === null) {
    throw new Error("frame must be a JSON object");
  }
  const { code, vars } = msg as { code?: unknown; vars?: unknown };
  if (typeof code !== "string") throw new Error("missing 'code'");
  const raw = vars ?? {};
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("'vars' must be an object");
  }
  const out: Record<string, RuntimeValue> = {};
  for (const [name, v] of Object.entries(raw)) {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(name)) {
      throw new Error(`invalid variable name '${name}'`);
    }
    out[name] = decodeValue(name, v, payload);
  }
  return { code, vars: out };
}

function decodeValue(
  name: string,
  v: unknown,
  payload: Uint8Array | null
): RuntimeValue {
  if (typeof v === "number") return RTV.num(v);
  if (typeof v === "boolean") return RTV.logical(v);
  if (typeof v === "string") return RTV.char(v);
  if (typeof v !== "object" || v === null || Array.isArray(v)) {
    throw new Error(`'${name}': unsupported value`);
  }
  const spec = v as {
    shape?: unknown;
    data?: unknown;
    dtype?: unknown;
    offset?: unknown;
  };
  const shape = spec.shape;
  if (
    !Array.isArray(shape) ||
    shape.length < 2 ||
    !shape.every(d => Number.isInteger(d) && d >= 0)
  ) {
    throw new Error(`'${name}': shape must be a list of >= 2 sizes`);
  }
  const n = (shape as number[]).reduce((a, b) => a * b, 1);

  if (spec.data !== undefined) {
    const data = spec.data;
    if (!Array.isArray(data) || data.length !== n) {
      throw new Error(`'${name}': data must have ${n} elements`);
    }
    const arr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = data[i];
      if (x !== null && typeof x !== "number") {
        throw new Error(`'${name}': data must contain numbers or null`);
      }
      arr[i] = x === null ? NaN : x;
    }
    return RTV.tensor(arr, shape as number[]);
  }

  if (payload === null) {
    throw new Error(`'${name}': binary arrays need a binary frame`);
  }
  const dtype = spec.dtype ?? "float64";
  if (dtype !== "float64" && dtype !== "float32") {
    throw new Error(`'${name}': dtype must be float64 or float32`);
  }
  const offset = spec.offset;
  if (!Number.isInteger(offset) || (offset as number) < 0) {
    throw new Error(`'${name}': offset must be a non-negative integer`);
  }
  const start = offset as number;
  const end = start + n * ITEM_SIZE[dtype as Dtype];
  if (end > payload.length) {
    throw new Error(`'${name}': array extends past the end of the payload`);
  }
  // Copy into a fresh, aligned buffer. (Not slice(): on a Node Buffer it
  // returns a view into a shared pool.)
  const bytes = new Uint8Array(payload.subarray(start, end));
  const arr =
    dtype === "float64"
      ? new Float64Array(bytes.buffer)
      : Float64Array.from(new Float32Array(bytes.buffer));
  return RTV.tensor(arr, shape as number[]);
}
