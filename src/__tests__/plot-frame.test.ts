import { describe, it, expect } from "vitest";
import { decodeFrame, isBinaryFrame } from "../plotFrame.js";
import type { RuntimeValue } from "../numbl-core/runtime/index.js";

const enc = (s: string) => new TextEncoder().encode(s);

/** Build a binary frame from a header object and payload bytes. */
function binaryFrame(header: object, payload: Uint8Array): Uint8Array {
  const h = enc(JSON.stringify(header));
  const out = new Uint8Array(8 + h.length + payload.length);
  out.set(enc("NBF1"), 0);
  new DataView(out.buffer).setUint32(4, h.length, true);
  out.set(h, 8);
  out.set(payload, 8 + h.length);
  return out;
}

const tensor = (v: RuntimeValue) =>
  v as unknown as { data: Float64Array; shape: number[] };

describe("plot frames", () => {
  it("decodes a JSON frame with scalars, strings and arrays", () => {
    const f = decodeFrame(
      enc(
        JSON.stringify({
          code: "plot(x)",
          vars: {
            t: 0.5,
            flag: true,
            name: "abc",
            x: { shape: [1, 3], data: [1, null, 3] },
          },
        })
      )
    );
    expect(f.code).toBe("plot(x)");
    expect(f.vars.t).toBe(0.5);
    const x = tensor(f.vars.x);
    expect(x.shape).toEqual([1, 3]);
    expect(Array.from(x.data)).toEqual([1, NaN, 3]);
  });

  it("decodes float64 and float32 arrays from a binary frame", () => {
    const a = new Float64Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([0.5, -1.5]);
    // Odd header length and an unaligned float64 offset are both allowed.
    const payload = new Uint8Array(3 + a.byteLength + b.byteLength);
    payload.set(new Uint8Array(a.buffer), 3);
    payload.set(new Uint8Array(b.buffer), 3 + a.byteLength);
    const f = decodeFrame(
      binaryFrame(
        {
          code: "imagesc(u)",
          vars: {
            u: { shape: [2, 3], dtype: "float64", offset: 3 },
            v: { shape: [2, 1], dtype: "float32", offset: 3 + a.byteLength },
            t: 1,
          },
        },
        payload
      )
    );
    expect(Array.from(tensor(f.vars.u).data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(tensor(f.vars.u).shape).toEqual([2, 3]);
    expect(Array.from(tensor(f.vars.v).data)).toEqual([0.5, -1.5]);
    expect(f.vars.t).toBe(1);
  });

  it("decodes a binary frame held in a pooled Node Buffer", () => {
    const a = new Float64Array([7, 8]);
    const frame = binaryFrame(
      { code: "", vars: { u: { shape: [1, 2], offset: 0 } } },
      new Uint8Array(a.buffer)
    );
    // Buffer.from(array) allocates from the shared pool, at a nonzero offset.
    const buf = Buffer.from(Array.from(frame));
    expect(buf.buffer.byteLength).toBeGreaterThan(buf.length);
    expect(Array.from(tensor(decodeFrame(buf).vars.u).data)).toEqual([7, 8]);
  });

  it("recognizes binary frames by their magic bytes", () => {
    expect(isBinaryFrame(binaryFrame({ code: "" }, new Uint8Array()))).toBe(
      true
    );
    expect(isBinaryFrame(enc('{"code":""}'))).toBe(false);
  });

  it("rejects malformed frames", () => {
    const bad =
      (header: object, payload = new Uint8Array(8)) =>
      () =>
        decodeFrame(binaryFrame(header, payload));
    expect(bad({ vars: {} })).toThrow(/missing 'code'/);
    expect(bad({ code: "", vars: { "1x": 1 } })).toThrow(/invalid variable/);
    expect(
      bad({ code: "", vars: { u: { shape: [2, 1], offset: 0 } } })
    ).toThrow(/past the end/);
    expect(
      bad({
        code: "",
        vars: { u: { shape: [1, 1], dtype: "int8", offset: 0 } },
      })
    ).toThrow(/dtype/);
    expect(() =>
      decodeFrame(
        enc(
          JSON.stringify({
            code: "",
            vars: { u: { shape: [1, 1], offset: 0 } },
          })
        )
      )
    ).toThrow(/binary frame/);
    const truncated = binaryFrame({ code: "" }, new Uint8Array());
    new DataView(truncated.buffer).setUint32(4, 1000, true);
    expect(() => decodeFrame(truncated)).toThrow(/header length/);
  });
});
