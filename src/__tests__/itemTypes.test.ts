import { describe, it, expect } from "vitest";
import {
  IType,
  typeToString,
  ItemType,
} from "../numbl-core/lowering/itemTypes";

describe("IType.unify", () => {
  // Basic / identity cases
  it("returns b when a is undefined", () => {
    expect(IType.unify(undefined, IType.Num)).toBe(IType.Num);
  });

  it("returns a when b is undefined", () => {
    expect(IType.unify(IType.Num, undefined)).toBe(IType.Num);
  });

  it("returns Unknown when either is Unknown", () => {
    expect(IType.unify(IType.Unknown, IType.Num)).toBe(IType.Unknown);
    expect(IType.unify(IType.Num, IType.Unknown)).toBe(IType.Unknown);
  });

  it("unifies identical simple types", () => {
    expect(IType.unify(IType.Num, IType.Num)).toBe(IType.Num);
    expect(IType.unify(IType.Bool, IType.Bool)).toBe(IType.Bool);
    expect(IType.unify(IType.Char, IType.Char)).toBe(IType.Char);
    expect(IType.unify(IType.String, IType.String)).toBe(IType.String);
    expect(IType.unify(IType.Void, IType.Void)).toBe(IType.Void);
  });

  it("unifies Num + Complex → Complex", () => {
    expect(IType.unify(IType.Num, IType.Complex)).toBe(IType.Complex);
    expect(IType.unify(IType.Complex, IType.Num)).toBe(IType.Complex);
  });

  it("returns Unknown for incompatible types", () => {
    expect(IType.unify(IType.Num, IType.String)).toBe(IType.Unknown);
    expect(IType.unify(IType.Bool, IType.Char)).toBe(IType.Unknown);
  });

  // Tensor
  it("unifies tensors — complex wins", () => {
    const a = IType.tensor({ isComplex: false });
    const b = IType.tensor({ isComplex: true });
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Tensor" }>;
    expect(r.kind).toBe("Tensor");
    expect(r.isComplex).toBe(true);
  });

  it("unifies tensors — logical only if both logical", () => {
    const a = IType.tensor({ isLogical: true });
    const b = IType.tensor({ isLogical: false });
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Tensor" }>;
    expect(r.isLogical).toBeFalsy();

    const c = IType.tensor({ isLogical: true });
    const d = IType.tensor({ isLogical: true });
    const r2 = IType.unify(c, d) as Extract<ItemType, { kind: "Tensor" }>;
    expect(r2.isLogical).toBe(true);
  });

  // Cell
  it("unifies cells with same element type", () => {
    const a = IType.cell(IType.Num, 3);
    const b = IType.cell(IType.Num, 3);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Cell" }>;
    expect(r.kind).toBe("Cell");
    expect(r.length).toBe(3);
  });

  it("unifies cells — unknown element absorbs known", () => {
    const a = IType.cell("unknown", 2);
    const b = IType.cell(IType.Num, 2);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Cell" }>;
    expect(r.kind).toBe("Cell");
    expect(typeToString(r.elementType as ItemType)).toBe("Number");
  });

  it("unifies cells — different lengths become unknown", () => {
    const a = IType.cell(IType.Num, 2);
    const b = IType.cell(IType.Num, 5);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Cell" }>;
    expect(r.length).toBe("unknown");
  });

  it("unifies cells — heterogeneous element types are recursively unified", () => {
    const a = IType.cell(IType.Num, "unknown");
    const b = IType.cell(IType.Complex, "unknown");
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Cell" }>;
    expect(r.kind).toBe("Cell");
    // Num + Complex → Complex
    expect(typeToString(r.elementType as ItemType)).toBe("ComplexNumber");
  });

  // Function
  it("unifies functions with matching arity", () => {
    const a = IType.func([IType.Num], IType.Num);
    const b = IType.func([IType.Num], IType.Num);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Function" }>;
    expect(r.kind).toBe("Function");
    expect(r.params.length).toBe(1);
  });

  it("returns Union for functions with different arity", () => {
    const a = IType.func([IType.Num], IType.Num);
    const b = IType.func([IType.Num, IType.Num], IType.Num);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Union" }>;
    expect(r.kind).toBe("Union");
    expect(r.types.length).toBe(2);
  });

  // Struct
  it("unifies structs — keeps intersection of fields", () => {
    const a = IType.struct(["a", "b", "c"]);
    const b = IType.struct(["a", "b", "d"]);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Struct" }>;
    expect(r.kind).toBe("Struct");
    expect(r.knownFields).toEqual(["a", "b"]);
  });

  it("unifies structs — identical fields preserved", () => {
    const a = IType.struct(["x", "y"]);
    const b = IType.struct(["x", "y"]);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Struct" }>;
    expect(r.knownFields).toEqual(["x", "y"]);
  });

  // Union — structural dedup
  it("unifies unions — deduplicates structurally equal types", () => {
    const a = IType.union([IType.num(), IType.Bool]);
    const b = IType.union([IType.num(), IType.String]);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Union" }>;
    expect(r.kind).toBe("Union");
    // Should have Number, Boolean, String — Number appears once despite different object refs
    const typeStrs = r.types.map(typeToString);
    expect(typeStrs).toEqual(["Number", "Boolean", "String"]);
  });

  it("unifies unions — keeps all distinct types", () => {
    const a = IType.union([IType.Num, IType.Bool]);
    const b = IType.union([IType.Char, IType.String]);
    const r = IType.unify(a, b) as Extract<ItemType, { kind: "Union" }>;
    expect(r.types.length).toBe(4);
  });

  // ClassInstance
  it("unifies matching ClassInstance types", () => {
    const a: ItemType = { kind: "ClassInstance", className: "Foo" };
    const b: ItemType = { kind: "ClassInstance", className: "Foo" };
    expect(IType.unify(a, b)).toEqual(a);
  });

  it("returns Unknown for different ClassInstance types", () => {
    const a: ItemType = { kind: "ClassInstance", className: "Foo" };
    const b: ItemType = { kind: "ClassInstance", className: "Bar" };
    expect(IType.unify(a, b)).toBe(IType.Unknown);
  });
});
