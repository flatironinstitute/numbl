// JS sibling of `isequal.h`. Same numeric equality semantics for the
// JS-JIT. Tensors carry `shape` / `data` / `imag` (the JS analogue of the
// C struct's `dims` / `real` / `imag`); a real tensor has no `imag`.

export function jit_isequal_tt(a, b) {
  if (a.shape.length !== b.shape.length) return 0;
  let n = 1;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return 0;
    n *= a.shape[i];
  }
  for (let i = 0; i < n; i++) {
    if (a.data[i] !== b.data[i]) return 0;
  }
  if (a.imag || b.imag) {
    for (let i = 0; i < n; i++) {
      if ((a.imag ? a.imag[i] : 0) !== (b.imag ? b.imag[i] : 0)) return 0;
    }
  }
  return 1;
}

export function jit_isequal_st(re, im, t) {
  let n = 1;
  for (let i = 0; i < t.shape.length; i++) n *= t.shape[i];
  if (n !== 1) return 0;
  return t.data[0] === re && (t.imag ? t.imag[0] : 0) === im ? 1 : 0;
}
