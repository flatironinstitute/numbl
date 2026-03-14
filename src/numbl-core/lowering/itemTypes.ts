/**
 * Type System for IR
 */

export type ItemType =
  | { kind: "Number" } // value is optional: if present, it's a known literal constant
  | { kind: "ComplexNumber" }
  | { kind: "Boolean" }
  | { kind: "Char" } // single-quoted char array: 'hello'
  | { kind: "String" } // double-quoted string: "hello"
  | {
      kind: "Tensor";
      isComplex?: boolean;
      isLogical?: boolean;
    }
  | {
      kind: "Cell";
      elementType: ItemType | "unknown";
      length: number | "unknown";
    }
  | { kind: "Function"; params: ItemType[]; returns: ItemType }
  | { kind: "Struct"; knownFields: string[] }
  | { kind: "Void" }
  | { kind: "Unknown" }
  | { kind: "Union"; types: ItemType[] }
  | { kind: "MultipleOutputs"; outputTypes: ItemType[] } // for return types of functions that return multiple outputs
  | { kind: "ClassInstance"; className: string }
  | { kind: "DummyHandle" };

export function typeToString(ty: ItemType): string {
  switch (ty.kind) {
    case "Number":
      return "Number";
    case "ComplexNumber":
      return "ComplexNumber";
    case "Boolean":
      return "Boolean";
    case "Char":
      return "Char";
    case "String":
      return "String";
    case "Tensor": {
      const complexStr = ty.isComplex ? ", complex" : ", real";
      const logicalStr = ty.isLogical ? ", logical" : "";
      return `Tensor<?${complexStr}${logicalStr}>`;
    }
    case "Cell": {
      const elementTypeStr =
        ty.elementType === "unknown" ? "?" : typeToString(ty.elementType);
      const lengthStr = ty.length === "unknown" ? "?" : ty.length.toString();
      return `Cell<elementType=${elementTypeStr}, length=${lengthStr}>`;
    }
    case "Function":
      return `Function<${ty.params.map(typeToString).join(", ")}, ${typeToString(
        ty.returns
      )}>`;
    case "Struct":
      return `Struct<${ty.knownFields ? ty.knownFields.join(", ") : "?"}>`;
    case "Void":
      return "Void";
    case "Unknown":
      return "Unknown";
    case "Union":
      return `Union<${ty.types.map(typeToString).join(" | ")}>`;
    case "ClassInstance":
      return `ClassInstance<${ty.className}>`;
    case "DummyHandle":
      return "DummyHandle";
    case "MultipleOutputs":
      return `MultipleOutputs<${ty.outputTypes.map(typeToString).join(", ")}>`;
    default:
      return "Unknown";
  }
}

// ── Query helpers ────────────────────────────────────────────────────────────

/** Returns true if the type is definitely Num, false if definitely not, undefined if unknown. */
export function isNum(t: ItemType): boolean | undefined {
  if (t.kind === "Number") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is definitely Tensor, false if definitely not, undefined if unknown. */
export function isTensor(t: ItemType): boolean | undefined {
  if (t.kind === "Tensor") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is definitely Complex, false if definitely not, undefined if unknown. */
export function isComplex(t: ItemType): boolean | undefined {
  if (t.kind === "ComplexNumber") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is fully unknown ({kind: "Unknown"}). */
export function isFullyUnknown(t: ItemType): boolean {
  return t.kind === "Unknown";
}

/** Returns true if the type is definitely String, false if definitely not, undefined if unknown. */
export function isString(t: ItemType): boolean | undefined {
  if (t.kind === "String") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is definitely Char, false if definitely not, undefined if unknown. */
export function isChar(t: ItemType): boolean | undefined {
  if (t.kind === "Char") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is definitely a Logical scalar, false if definitely not, undefined if unknown. */
export function isLogicalScalar(t: ItemType): boolean | undefined {
  if (t.kind === "Boolean") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is definitely Void, false if definitely not, undefined if unknown. */
export function isVoid(t: ItemType): boolean | undefined {
  if (t.kind === "Void") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is definitely a Cell, false if definitely not, undefined if unknown. */
export function isCell(t: ItemType): boolean | undefined {
  if (t.kind === "Cell") return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** Returns true if the type is a scalar numeric (Num, Complex, or Logical), false if not, undefined if unknown. */
export function isScalarNumeric(t: ItemType): boolean | undefined {
  if (t.kind === "Number" || t.kind === "ComplexNumber" || t.kind === "Boolean")
    return true;
  if (t.kind === "Unknown") return undefined;
  return false;
}

/** True for scalar numeric types (Number, Boolean, ComplexNumber). */
export const isScalarType = (t: ItemType): boolean =>
  t.kind === "Number" || t.kind === "Boolean" || t.kind === "ComplexNumber";

/** True for complex-valued types (ComplexNumber or complex Tensor). */
export const isComplexType = (t: ItemType): boolean =>
  t.kind === "ComplexNumber" || (t.kind === "Tensor" && !!t.isComplex);

export const IType = {
  Num: { kind: "Number" } as ItemType,
  Complex: { kind: "ComplexNumber" } as ItemType,
  Bool: { kind: "Boolean" } as ItemType,
  Logical: { kind: "Boolean" } as ItemType,
  Char: { kind: "Char" } as ItemType,
  String: { kind: "String" } as ItemType,
  Void: { kind: "Void" } as ItemType,
  Unknown: { kind: "Unknown" } as ItemType,
  DummyHandle: { kind: "DummyHandle" } as ItemType,

  /** Create a Num type */
  num(): ItemType {
    return { kind: "Number" };
  },

  tensor(
    opts: {
      isComplex?: boolean;
      isLogical?: boolean;
    } = {}
  ): ItemType {
    return {
      kind: "Tensor",
      isComplex: opts.isComplex,
      isLogical: opts.isLogical,
    };
  },

  cell(
    elementType: ItemType | "unknown" = "unknown",
    length: number | "unknown" = "unknown"
  ): ItemType {
    return { kind: "Cell", elementType, length };
  },

  func(params: ItemType[], returns: ItemType): ItemType {
    return { kind: "Function", params, returns };
  },

  struct(fields: string[]): ItemType {
    return { kind: "Struct", knownFields: fields };
  },

  union(types: ItemType[]): ItemType {
    return { kind: "Union", types };
  },

  /** Unify two types, producing a common supertype */
  // undefined means it hasn't been assigned yet, unknown means it has been assigned and is unknown
  unify(
    a: ItemType | undefined,
    b: ItemType | undefined
  ): ItemType | undefined {
    if (a === undefined) {
      return b;
    }
    if (b === undefined) {
      return a;
    }
    if (a.kind === "Unknown" || b.kind === "Unknown") {
      return IType.Unknown;
    }
    // Complex + Num or Num + Complex → Complex
    if (
      (a.kind === "ComplexNumber" && b.kind === "Number") ||
      (a.kind === "Number" && b.kind === "ComplexNumber")
    ) {
      return IType.Complex;
    }
    if (a.kind === b.kind) {
      switch (a.kind) {
        case "Number":
        case "ComplexNumber":
        case "Boolean":
        case "Char":
        case "String":
        case "Void":
        case "DummyHandle":
          return a;
        case "Tensor": {
          if (b.kind === "Tensor") {
            const isComplex = a.isComplex || b.isComplex || undefined;
            const isLogical = (a.isLogical && b.isLogical) || undefined;
            return { kind: "Tensor", isComplex, isLogical };
          }
          break;
        }
        case "Cell": {
          if (b.kind === "Cell") {
            // TODO: check this carefully
            const elementType =
              a.elementType === b.elementType
                ? a.elementType
                : a.elementType === "unknown"
                  ? b.elementType
                  : b.elementType === "unknown"
                    ? a.elementType
                    : IType.unify(a.elementType, b.elementType);
            const length = a.length === b.length ? a.length : "unknown";
            const elementType0 =
              elementType === undefined ? IType.Unknown : elementType;
            return { kind: "Cell", elementType: elementType0, length };
          }
          break;
        }
        case "Function": {
          // TODO: check this carefully
          if (b.kind === "Function") {
            if (a.params.length !== b.params.length) {
              return IType.Unknown;
            }
            const params = a.params.map((p, i) => IType.unify(p, b.params[i]));
            const returns = IType.unify(a.returns, b.returns);
            const params0 = params.map(p =>
              p === undefined ? IType.Unknown : p
            );
            const returns0 = returns === undefined ? IType.Unknown : returns;
            return { kind: "Function", params: params0, returns: returns0 };
          }
          break;
        }
        case "Struct": {
          // TODO: check this carefully
          if (b.kind === "Struct") {
            const knownFields = a.knownFields.filter(f =>
              b.knownFields.includes(f)
            );
            return { kind: "Struct", knownFields };
          }
          break;
        }
        case "Union": {
          // TODO: check this carefully
          if (b.kind === "Union") {
            // Deduplicate by structural equality (typeToString), not reference
            const seen = new Set<string>();
            const types: ItemType[] = [];
            for (const t of [...a.types, ...b.types]) {
              const key = typeToString(t);
              if (!seen.has(key)) {
                seen.add(key);
                types.push(t);
              }
            }
            return { kind: "Union", types };
          }
          break;
        }
        case "ClassInstance": {
          if (b.kind === "ClassInstance" && a.className === b.className) {
            return a;
          }
          break;
        }
      }
    }
    return IType.Unknown;
  },
};
