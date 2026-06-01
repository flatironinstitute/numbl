/**
 * Runtime value carrier for the interpreter backend. Mirrors nexl's
 * `RuntimeValue` design: a discriminated-via-typeof union for the
 * common cases plus tagged-object wrappers for the kinds JS can't
 * distinguish natively.
 *
 * MVP scope (Phase 3): real-double scalars, booleans (logical),
 * strings (double-quoted), chars (single-quoted), real-double tensors.
 * Complex / structs / classes / handles / void / cells follow as the
 * type lattice and interpreter coverage expand.
 *
 * Builtin `call` implementations narrow via `typeof` / `isTensor` /
 * `isChar` and dispatch through the same `argTypes` shape that
 * `emitC` / `emitJs` consume — keeping the three backends parallel.
 */

export type RuntimeValue =
  | number
  | boolean
  | string
  | RuntimeTensor
  | RuntimeChar
  | RuntimeComplex
  | RuntimeStruct
  | RuntimeHandle;

/** Function handle. Named handles (`@foo`) keep the source name and
 *  dispatch via the interpreter's normal call-site resolution at the
 *  call point. Anonymous handles (`@(p1, ...) body`) capture every
 *  visible binding by value at the @-site and run `body` in a fresh
 *  env when invoked. */
export type RuntimeHandle =
  | {
      readonly mtoc2Handle: true;
      readonly kind: "named";
      readonly name: string;
    }
  | {
      readonly mtoc2Handle: true;
      readonly kind: "anon";
      readonly params: ReadonlyArray<string>;
      readonly body: unknown;
      readonly captures: Readonly<Record<string, RuntimeValue>>;
    };

/** Narrow a RuntimeValue to a function handle. */
export function isHandleValue(v: RuntimeValue): v is RuntimeHandle {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeHandle).mtoc2Handle === true
  );
}

/** Struct/object — plain JS object whose keys are field names. The
 *  c-aot path emits a typedef'd C struct; the JS path just uses a
 *  bare object. Recurses through field values that may themselves be
 *  any RuntimeValue (including nested structs). */
export type RuntimeStruct = { readonly [field: string]: RuntimeValue };

/** Real or complex double tensor. `data` is the real lane,
 *  column-major to match numbl's `RuntimeTensor.data`. `imag` is the
 *  imaginary lane (same length and orientation as `data`); absent for
 *  real-only tensors. Mirrors the C representation's split-buffer
 *  layout — codegen / interpreter dispatch on `imag !== undefined`
 *  the way the C side dispatches on `imag != NULL`. The shape array
 *  is owned by the value; do not mutate.
 *
 *  `isLogical` flags this tensor as the result of a logical operation
 *  (`~`, `<`, `==`, `&&`, …) so downstream index-slot resolution can
 *  treat it as a mask instead of an IndexVec — same role as numbl's
 *  `_isLogical` on its RuntimeTensor. The flag is one-way (once set,
 *  stays set across copies via the helper) and absent on numeric
 *  tensors; helper sites that don't care leave it undefined. */
export interface RuntimeTensor {
  readonly mtoc2Tag: "tensor";
  readonly shape: number[];
  readonly data: Float64Array;
  readonly imag?: Float64Array;
  readonly isLogical?: boolean;
}

/** Single-quoted char-array (`'foo'`). Distinct from `"foo"` strings
 *  because numbl treats them differently: `length('foo') === 3` but
 *  `length("foo") === 1`. */
export interface RuntimeChar {
  readonly mtoc2Tag: "char";
  readonly value: string;
}

/** Scalar complex value — JS-side analogue of the `mtoc2_complex_t`
 *  struct in `cscalar.h`. The cscalar runtime helpers
 *  (`mtoc2_cmake`, `mtoc2_creal`, etc.) consume and produce this
 *  shape. We DON'T tag it the way tensor / char are tagged because
 *  emitJs literals (`{re: 0, im: 1}`) and runtime calls produce the
 *  bare object form — adding a tag would force every helper to
 *  re-wrap. Use `isComplexValue` for narrowing. */
export interface RuntimeComplex {
  readonly re: number;
  readonly im: number;
}

/** Narrow a RuntimeValue to a complex scalar. Matches the cscalar
 *  helpers' `{re, im}` shape. */
export function isComplexValue(v: RuntimeValue): v is RuntimeComplex {
  return (
    typeof v === "object" &&
    v !== null &&
    !isTensor(v) &&
    !isChar(v) &&
    typeof (v as RuntimeComplex).re === "number" &&
    typeof (v as RuntimeComplex).im === "number"
  );
}

export function makeTensor(shape: number[], data: Float64Array): RuntimeTensor {
  let total = 1;
  for (const s of shape) total *= s;
  if (data.length !== total) {
    throw new Error(
      `makeTensor: shape [${shape.join(",")}] requires ${total} elements, got ${data.length}`
    );
  }
  return { mtoc2Tag: "tensor", shape: shape.slice(), data };
}

/** Complex tensor with both lanes. `data` is the real lane, `imag`
 *  the imaginary lane; both must have the same length and match the
 *  product of `shape`. */
export function makeComplexTensor(
  shape: number[],
  data: Float64Array,
  imag: Float64Array
): RuntimeTensor {
  let total = 1;
  for (const s of shape) total *= s;
  if (data.length !== total || imag.length !== total) {
    throw new Error(
      `makeComplexTensor: shape [${shape.join(",")}] requires ${total} elements, got ${data.length}/${imag.length}`
    );
  }
  return { mtoc2Tag: "tensor", shape: shape.slice(), data, imag };
}

/** True iff this tensor carries an imaginary lane. */
export function isComplexTensor(v: RuntimeTensor): boolean {
  return v.imag !== undefined;
}

export function makeChar(value: string): RuntimeChar {
  return { mtoc2Tag: "char", value };
}

export function isTensor(v: RuntimeValue): v is RuntimeTensor {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeTensor).mtoc2Tag === "tensor"
  );
}

export function isChar(v: RuntimeValue): v is RuntimeChar {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as RuntimeChar).mtoc2Tag === "char"
  );
}

/** Materialize a scalar value as a plain JS number. Used by builtins
 *  that need to treat scalar tensors and scalar numbers uniformly. */
export function toScalarNumber(v: RuntimeValue): number {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (isTensor(v) && v.data.length === 1) return v.data[0];
  throw new Error("toScalarNumber: value is not a scalar number");
}

/** MATLAB-style truthiness — for `if` / `while` / `&&` / `||`.
 *  Scalar non-zero, all-elements-non-zero for tensors, nonempty
 *  text. The interpreter uses this; codegen relies on the matching
 *  C-side `__nonzero` helpers. */
export function isTruthy(v: RuntimeValue): boolean {
  if (typeof v === "number") return v !== 0;
  if (typeof v === "boolean") return v;
  if (typeof v === "string") return v.length > 0;
  if (isChar(v)) return v.value.length > 0;
  if (isComplexValue(v)) return v.re !== 0 || v.im !== 0;
  if (isTensor(v)) {
    if (v.data.length === 0) return false;
    const im = v.imag;
    for (let i = 0; i < v.data.length; i++) {
      const reZero = v.data[i] === 0;
      const imZero = im === undefined || im[i] === 0;
      if (reZero && imZero) return false;
    }
    return true;
  }
  return false;
}
