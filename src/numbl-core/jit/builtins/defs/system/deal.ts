/**
 * `deal` — multi-output passthrough.
 *
 *   [a, b, c] = deal(x, y, z)   → a=x, b=y, c=z       (N-arg form)
 *   [a, b, c] = deal(x)         → a=x, b=x, c=x       (broadcast form)
 *   a        = deal(x)          → x
 *
 * Numbl's rule (the dialect oracle): single-input broadcasts to any
 * `nargout >= 1`; otherwise `args.length` must equal `nargout` or
 * it's a runtime error. mtoc2 lifts the equality check into
 * `transfer` so the mismatch is a type-time error.
 *
 * The result types are the input types as-is. The c-aot backend
 * still has to emit `_copy(...)` for owned types to satisfy the
 * "freshly-owned at every consume site" invariant; the js-aot
 * backend uses `mtoc2_deep_clone` for the same reason; the
 * interpreter follows numbl and deep-clones owned values too.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  cellTypedefName,
  classTypedefName,
  handleTypedefName,
  isOwned,
  isScalarRealNumeric,
  structTypedefName,
  type Type,
  typeToString,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import type { RuntimeValue } from "../../../runtime/value.js";
import { mtoc2_deep_clone } from "../../runtime/snippets.gen.js";

/** Owned-helper-name family for a type. Returns `null` for scalar /
 *  non-owned. Mirrors `src/codegen/cHelpers.ts:ownedHelpersFor` but
 *  inlined here to avoid a builtin → codegen import. The names are
 *  stable contracts: the runtime snippets (`tensor`, `string`,
 *  `char`) and `emitNamedTypedef` (struct / class / handle / cell)
 *  expose them in the emitted C. */
function ownedHelpers(
  t: Type
): { copy: string; assign: string; cType: string; isRuntime: boolean } | null {
  if (
    t.kind === "Numeric" &&
    t.dims.some(d => d.kind !== "exact" || d.value !== 1)
  ) {
    // Multi-element tensor.
    const isComplex = t.isComplex;
    return {
      copy: isComplex ? "mtoc2_tensor_copy_complex" : "mtoc2_tensor_copy",
      assign: "mtoc2_tensor_assign",
      cType: "mtoc2_tensor_t",
      isRuntime: true,
    };
  }
  if (t.kind === "String") {
    return {
      copy: "mtoc2_string_copy",
      assign: "mtoc2_string_assign",
      cType: "mtoc2_string_t",
      isRuntime: true,
    };
  }
  if (t.kind === "Char") {
    return {
      copy: "mtoc2_char_tensor_copy",
      assign: "mtoc2_char_tensor_assign",
      cType: "mtoc2_char_tensor_t",
      isRuntime: true,
    };
  }
  if (t.kind === "Struct") {
    const name = structTypedefName(t);
    return {
      copy: `${name}_copy`,
      assign: `${name}_assign`,
      cType: name,
      isRuntime: false,
    };
  }
  if (t.kind === "Class") {
    const name = classTypedefName(t);
    return {
      copy: `${name}_copy`,
      assign: `${name}_assign`,
      cType: name,
      isRuntime: false,
    };
  }
  if (t.kind === "Handle") {
    const name = handleTypedefName(t);
    return {
      copy: `${name}_copy`,
      assign: `${name}_assign`,
      cType: name,
      isRuntime: false,
    };
  }
  if (t.kind === "Cell") {
    const name = cellTypedefName(t);
    return {
      copy: `${name}_copy`,
      assign: `${name}_assign`,
      cType: name,
      isRuntime: false,
    };
  }
  return null;
}

/** Validate args/nargout and return the source-arg index for each
 *  output slot. Throws on shape mismatches. */
function dealSlotIndices(argTypes: Type[], nargout: number): number[] {
  if (argTypes.length === 0) {
    throw new TypeError("'deal' requires at least one input argument");
  }
  if (nargout < 1) {
    throw new UnsupportedConstruct(
      `'deal' requires at least one output (got nargout=${nargout})`
    );
  }
  if (argTypes.length === 1) {
    // Broadcast form: every output reads arg 0.
    return new Array(nargout).fill(0);
  }
  if (argTypes.length !== nargout) {
    throw new TypeError(
      `'deal': number of inputs (${argTypes.length}) must match number of outputs (${nargout})`
    );
  }
  return argTypes.map((_, i) => i);
}

/** Reject types that can't appear in a multi-output slot. Scalar
 *  real numeric and owned types are fine; scalar complex is the
 *  notable hole. Pre-validate so the user gets a deal-attributed
 *  error rather than the generic multi-assign rejection. */
function rejectUnsupportedSlot(t: Type, argIdx: number): void {
  if (isScalarRealNumeric(t)) return;
  if (isOwned(t)) return;
  throw new UnsupportedConstruct(
    `'deal' input ${argIdx + 1} has type ${typeToString(t)}; ` +
      `this type isn't supported in a multi-output slot`
  );
}

export const deal: Builtin = {
  name: "deal",

  transfer(argTypes, nargout) {
    const slotIdx = dealSlotIndices(argTypes, nargout);
    if (nargout >= 2) {
      for (let i = 0; i < nargout; i++) {
        rejectUnsupportedSlot(argTypes[slotIdx[i]], slotIdx[i]);
      }
    }
    return slotIdx.map(i => argTypes[i]);
  },

  emitC({ argsC, argTypes, nargout, outArgsC, useRuntime }) {
    const slotIdx = dealSlotIndices(argTypes, nargout);

    // `outArgsC !== undefined` ⇔ called from MultiAssignCall (the
    // framework only builds the out-pointer list there). It wraps our
    // emission in a `{ }` block, so we may declare temporaries.
    if (outArgsC !== undefined) {
      if (nargout >= 2) {
        // With ≥2 outputs an output may alias a later source — the swap
        // idiom `[a, b] = deal(b, a)`. Writing in source order without
        // temporaries makes the second source read the value the first
        // write already stored (and, for owned types, `_assign` frees
        // the old buffer first → use-after-free). Snapshot EVERY source
        // into a temporary first (the owned snapshot is a fresh `_copy`),
        // then write all outputs from the temporaries. This matches the
        // JS path (array literal built before destructuring) and the
        // interpreter (reads all args before assigning).
        const decls: string[] = [];
        const writes: string[] = [];
        for (let i = 0; i < nargout; i++) {
          const srcTy = argTypes[slotIdx[i]];
          const argC = argsC[slotIdx[i]];
          const h = ownedHelpers(srcTy);
          const tmp = `_mtoc2_deal_t${i}`;
          if (h !== null) {
            if (h.isRuntime) {
              useRuntime(h.copy);
              useRuntime(h.assign);
            }
            decls.push(`${h.cType} ${tmp} = ${h.copy}(${argC});`);
            writes.push(`${h.assign}(${outArgsC[i]}, ${tmp})`);
          } else {
            // nargout ≥ 2 slots are scalar-real (transfer rejects scalar
            // complex / unsupported types), so a `double` temp is exact.
            decls.push(`double ${tmp} = ${argC};`);
            writes.push(`(*${outArgsC[i]} = ${tmp})`);
          }
        }
        return `${decls.join(" ")} (void)(${writes.join(", ")})`;
      }
      // nargout === 1 (`[a] = deal(x)`): one output, one source — no
      // aliasing is possible, so write through the pointer directly.
      const srcTy = argTypes[slotIdx[0]];
      const argC = argsC[slotIdx[0]];
      const h = ownedHelpers(srcTy);
      if (h !== null) {
        if (h.isRuntime) {
          useRuntime(h.copy);
          useRuntime(h.assign);
        }
        return `((void)(${h.assign}(${outArgsC[0]}, ${h.copy}(${argC}))))`;
      }
      return `((void)(*${outArgsC[0]} = ${argC}))`;
    }

    // Value position (`a = deal(x)` / `disp(deal(x))`): return an
    // expression. Owned types wrap in `_copy` so the result is a
    // freshly-owned tensor per mtoc2's invariant; scalars pass
    // through bare.
    const srcTy = argTypes[slotIdx[0]];
    const h = ownedHelpers(srcTy);
    if (h !== null) {
      if (h.isRuntime) useRuntime(h.copy);
      return `${h.copy}(${argsC[slotIdx[0]]})`;
    }
    return argsC[slotIdx[0]];
  },

  emitJs({ argsJs, argTypes, nargout, outTargetsJs, useRuntime }) {
    const slotIdx = dealSlotIndices(argTypes, nargout);
    const cloneIfOwned = (i: number): string => {
      const srcTy = argTypes[slotIdx[i]];
      if (isOwned(srcTy)) {
        useRuntime("mtoc2_deep_clone");
        return `mtoc2_deep_clone(${argsJs[slotIdx[i]]})`;
      }
      return argsJs[slotIdx[i]];
    };
    // `outTargetsJs !== undefined` ⇔ called from MultiAssignCall —
    // the framework wraps our return in `[t1, ...] = <expr>;` so we
    // must return an array literal of the cloned values (one per
    // slot). Single-lvalue `[a] = deal(x)` lands here too.
    if (outTargetsJs !== undefined) {
      const parts: string[] = [];
      for (let i = 0; i < nargout; i++) parts.push(cloneIfOwned(i));
      return `[${parts.join(", ")}]`;
    }
    // Value position: return a single expression value.
    return cloneIfOwned(0);
  },

  call({ args, argTypes, nargout }) {
    const slotIdx = dealSlotIndices(argTypes, nargout);
    const cloneIfOwned = (i: number): RuntimeValue => {
      const v = args[slotIdx[i]];
      if (isOwned(argTypes[slotIdx[i]])) {
        return mtoc2_deep_clone(v) as RuntimeValue;
      }
      return v;
    };
    const out: RuntimeValue[] = [];
    for (let i = 0; i < nargout; i++) out.push(cloneIfOwned(i));
    return out;
  },
};
