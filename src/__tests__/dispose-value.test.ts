import { describe, it, expect, beforeEach } from "vitest";
import {
  allocFloat64,
  allocFloatX,
  getAllocStats,
  resetAllocStats,
  clearPool,
} from "../numbl-core/runtime/alloc.js";
import { disposeValue } from "../numbl-core/runtime/utils.js";
import { FloatXArray } from "../numbl-core/runtime/types.js";
import type {
  RuntimeTensor,
  RuntimeCell,
  RuntimeStruct,
  RuntimeStructArray,
  RuntimeSparseMatrix,
  RuntimeDictionary,
  RuntimeClassInstance,
  RuntimeClassInstanceArray,
  RuntimeFunction,
} from "../numbl-core/runtime/types.js";

const FLOATX_BYTES = (FloatXArray as unknown) === Float32Array ? 4 : 8;

function tensor(n: number, withImag = false): RuntimeTensor {
  const data = allocFloatX(n);
  data.fill(1);
  const t: RuntimeTensor = { kind: "tensor", data, shape: [n] };
  if (withImag) {
    const imag = allocFloatX(n);
    imag.fill(2);
    t.imag = imag;
  }
  return t;
}

describe("disposeValue", () => {
  beforeEach(() => {
    clearPool();
    resetAllocStats();
  });

  it("disposes a tensor's data buffer", () => {
    const t = tensor(8);
    disposeValue(t);
    const s = getAllocStats();
    expect(s.disposeCount).toBe(1);
    expect(s.disposeBytes).toBe(8 * FLOATX_BYTES);
    expect(s.poolBuffersHeld).toBe(1);
  });

  it("disposes both real and imag buffers of a complex tensor", () => {
    const t = tensor(4, true);
    disposeValue(t);
    const s = getAllocStats();
    expect(s.disposeCount).toBe(2);
    expect(s.disposeBytes).toBe(2 * 4 * FLOATX_BYTES);
    expect(s.poolBuffersHeld).toBe(2);
  });

  it("disposes nested tensors inside a cell", () => {
    const cell: RuntimeCell = {
      kind: "cell",
      data: [tensor(3), tensor(5), tensor(7)],
      shape: [1, 3],
    };
    disposeValue(cell);
    const s = getAllocStats();
    expect(s.disposeCount).toBe(3);
    expect(s.disposeBytes).toBe((3 + 5 + 7) * FLOATX_BYTES);
  });

  it("disposes tensor fields of a struct", () => {
    const s: RuntimeStruct = {
      kind: "struct",
      fields: new Map<string, RuntimeTensor>([
        ["a", tensor(2)],
        ["b", tensor(6)],
      ]),
    };
    disposeValue(s);
    const stats = getAllocStats();
    expect(stats.disposeCount).toBe(2);
    expect(stats.disposeBytes).toBe((2 + 6) * FLOATX_BYTES);
  });

  it("disposes tensors in a struct array", () => {
    const sa: RuntimeStructArray = {
      kind: "struct_array",
      fieldNames: ["t"],
      elements: [
        { kind: "struct", fields: new Map([["t", tensor(4)]]) },
        { kind: "struct", fields: new Map([["t", tensor(8)]]) },
      ],
    };
    disposeValue(sa);
    expect(getAllocStats().disposeCount).toBe(2);
    expect(getAllocStats().disposeBytes).toBe((4 + 8) * FLOATX_BYTES);
  });

  it("disposes pr and pi of a sparse matrix", () => {
    const pr = allocFloat64(5);
    pr.fill(1);
    const pi = allocFloat64(5);
    pi.fill(2);
    const sp: RuntimeSparseMatrix = {
      kind: "sparse_matrix",
      m: 10,
      n: 10,
      ir: new Int32Array([0, 2, 5, 7, 9]),
      jc: new Int32Array([0, 2, 3, 5]),
      pr,
      pi,
    };
    disposeValue(sp);
    const s = getAllocStats();
    expect(s.disposeCount).toBe(2);
    expect(s.disposeBytes).toBe(2 * 5 * 8);
  });

  it("disposes keys and values of a dictionary", () => {
    const d: RuntimeDictionary = {
      kind: "dictionary",
      entries: new Map([
        ["k1", { key: tensor(2), value: tensor(3) }],
        ["k2", { key: tensor(4), value: tensor(5) }],
      ]),
    };
    disposeValue(d);
    expect(getAllocStats().disposeCount).toBe(4);
  });

  it("disposes value-class instance fields", () => {
    const ci: RuntimeClassInstance = {
      kind: "class_instance",
      className: "Foo",
      fields: new Map<string, RuntimeTensor>([
        ["x", tensor(3)],
        ["y", tensor(4)],
      ]),
      isHandleClass: false,
    };
    disposeValue(ci);
    expect(getAllocStats().disposeCount).toBe(2);
  });

  it("does NOT dispose handle-class instance fields", () => {
    const ci: RuntimeClassInstance = {
      kind: "class_instance",
      className: "Foo",
      fields: new Map<string, RuntimeTensor>([["x", tensor(3)]]),
      isHandleClass: true,
    };
    disposeValue(ci);
    expect(getAllocStats().disposeCount).toBe(0);
  });

  it("does NOT dispose value class array of handle elements", () => {
    const cia: RuntimeClassInstanceArray = {
      kind: "class_instance_array",
      className: "H",
      elements: [
        {
          kind: "class_instance",
          className: "H",
          fields: new Map([["x", tensor(2)]]),
          isHandleClass: true,
        },
      ],
    };
    disposeValue(cia);
    expect(getAllocStats().disposeCount).toBe(0);
  });

  it("disposes value-class array of value-class elements", () => {
    const cia: RuntimeClassInstanceArray = {
      kind: "class_instance_array",
      className: "V",
      elements: [
        {
          kind: "class_instance",
          className: "V",
          fields: new Map([["x", tensor(2)]]),
          isHandleClass: false,
        },
        {
          kind: "class_instance",
          className: "V",
          fields: new Map([["x", tensor(3)]]),
          isHandleClass: false,
        },
      ],
    };
    disposeValue(cia);
    expect(getAllocStats().disposeCount).toBe(2);
  });

  it("ignores primitives and char/complex/function values", () => {
    disposeValue(42);
    disposeValue(true);
    disposeValue("hello");
    disposeValue({ kind: "char", value: "abc" });
    disposeValue({ kind: "complex_number", re: 1, im: 2 });
    const fn: RuntimeFunction = {
      kind: "function",
      name: "f",
      captures: [],
      impl: "user",
    };
    disposeValue(fn);
    expect(getAllocStats().disposeCount).toBe(0);
  });

  it("disposes deeply nested structures", () => {
    const inner: RuntimeStruct = {
      kind: "struct",
      fields: new Map<string, RuntimeCell>([
        [
          "items",
          {
            kind: "cell",
            data: [tensor(3), tensor(4)],
            shape: [1, 2],
          },
        ],
      ]),
    };
    const outer: RuntimeCell = {
      kind: "cell",
      data: [inner, tensor(5)],
      shape: [1, 2],
    };
    disposeValue(outer);
    expect(getAllocStats().disposeCount).toBe(3);
    expect(getAllocStats().disposeBytes).toBe((3 + 4 + 5) * FLOATX_BYTES);
  });
});
