/**
 * Coarse argument types used by function resolution (`resolveFunction`):
 * only `ClassInstance` (and its class name) affects which target a call
 * resolves to.
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
  | { kind: "Struct"; knownFields: Record<string, ItemType> }
  | { kind: "Void" }
  | { kind: "Unknown" }
  | { kind: "Union"; types: ItemType[] }
  | { kind: "MultipleOutputs"; outputTypes: ItemType[] } // for return types of functions that return multiple outputs
  | { kind: "ClassInstance"; className: string }
  | { kind: "DummyHandle" }
  | { kind: "SparseMatrix"; isComplex?: boolean }
  | { kind: "Dictionary" };
