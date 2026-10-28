/* @snippet name=jit_tensor_t */
/* mtoc tensor runtime: the single C representation for every
 * multi-element tensor mtoc emits.
 *
 * Storage mirrors numbl's split layout — `real` and `imag` are
 * separate Float64 buffers (not interleaved real/imag pairs). For a
 * statically-real tensor, `imag` is NULL: the type system tracks
 * `isComplex` at compile time, so codegen knows up front whether to
 * touch the imag side. There is no runtime branch on `imag != NULL`.
 *
 * Scalars never use this struct — real scalars are bare `double`,
 * complex scalars are `double _Complex`. Anything the type system
 * classifies as multi-element (any axis is not statically 1) gets
 * one `jit_tensor_t` value. The data buffers are column-major to match
 * MATLAB / LAPACK.
 *
 * Shape: `ndim` axes with sizes `dims[0..ndim-1]`. The minimum
 * logical ndim is 2 (matching numbl) — a row vector is `{1, n}` and
 * a column vector is `{n, 1}`. `JIT_MAX_NDIM` caps the inline dims
 * array; tensors with more axes than that are unrepresentable in
 * mtoc today (the static type system enforces this at lowering).
 * Keeping `dims` inline preserves the value-typed semantics of
 * `jit_tensor_t` — assign / copy / free do a struct copy and
 * touch only the two heap pointers; the shape rides along for free.
 *
 * Storage is heap-allocated via `jit_alloc` at every assignment
 * site. The struct is predeclared with `real = imag = NULL` and
 * `ndim = 0` (no dims populated); the first assignment populates
 * them, and subsequent reassignments at a different runtime shape
 * free the previous buffers and alloc fresh ones. `free` of the
 * predeclared NULLs is a no-op (well-defined by C), so the cleanup
 * path is uniform for first and subsequent assignments alike.
 *
 * The `JIT_RESTRICT` qualifier on the buffer pointers tells the
 * compiler that distinct `jit_tensor_t` values' buffers do not
 * alias each other. It expands to `__restrict__` under GCC/Clang
 * and to nothing on compilers that don't recognize it.
 */

#ifndef JIT_RESTRICT
# if defined(__GNUC__) || defined(__clang__)
#  define JIT_RESTRICT __restrict__
# else
#  define JIT_RESTRICT
# endif
#endif

#ifndef JIT_MAX_NDIM
#define JIT_MAX_NDIM 8
#endif

typedef struct {
  double *JIT_RESTRICT real;   /* always non-NULL */
  double *JIT_RESTRICT imag;   /* NULL iff the tensor is statically real */
  int  ndim;
  long dims[JIT_MAX_NDIM];
} jit_tensor_t;

/* OpenMP `#pragma omp parallel for if(n > 1024)` for elementwise
 * loops that iterate over a length-`n` index. Active only when the
 * C compiler runs with `-fopenmp` (which the JIT's build adds when
 * `--threads` is `auto` or a number `>= 2`); otherwise `_OPENMP` is
 * undefined and the macro expands to nothing — the loop stays
 * serial and the pragma never reaches the compiler. The
 * `if(n > 1024)` clause keeps small loops serial regardless, so
 * OpenMP region startup doesn't dominate for tensors that don't
 * amortize the overhead. Defined once here because every elementwise
 * runtime helper depends transitively on `jit_tensor_t`. */
#ifndef JIT_OMP_PARFOR_N
# ifdef _OPENMP
#  define JIT_OMP_PARFOR_N _Pragma("omp parallel for if(n > 1024)")
# else
#  define JIT_OMP_PARFOR_N
# endif
#endif
