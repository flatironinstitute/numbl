/**
 * Unified rich type information about runtime values and expressions.
 *
 * For now this is a re-export of `JitType` from the JIT module — that
 * type already covers most of what we need (scalar kind, complex/
 * logical, exact value retention, optional tensor shape). Future
 * enrichment (more aggressive shape tracking, exact-value retention
 * for strings/cells, struct field unification) lands here so executors
 * have one place to read from.
 *
 * `cacheKey` projects volatile bits out via per-executor logic; the
 * default key helper is `typeInfoKey`.
 */

export type { JitType as TypeInfo, SignCategory } from "../jit/jitTypes.js";
export {
  jitTypeKey as typeInfoKey,
  unifyJitTypes as unifyTypeInfo,
  isScalarType,
  isTensorType,
  isComplexType,
  isKnownInteger,
} from "../jit/jitTypes.js";
export { inferJitType as inferTypeInfo } from "../interpreter/builtins/types.js";
