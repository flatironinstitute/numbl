// JS sibling of `isequal.h`. Same real-numeric equality semantics for
// the js-aot backend. Tensors carry `shape` / `data` (the JS analogue
// of the C struct's `dims` / `real`).

export function mtoc2_isequal_tt(a, b) {
  if (a.shape.length !== b.shape.length) return 0;
  let n = 1;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return 0;
    n *= a.shape[i];
  }
  for (let i = 0; i < n; i++) {
    if (a.data[i] !== b.data[i]) return 0;
  }
  return 1;
}

export function mtoc2_isequal_st(s, t) {
  let n = 1;
  for (let i = 0; i < t.shape.length; i++) n *= t.shape[i];
  if (n !== 1) return 0;
  return t.data[0] === s ? 1 : 0;
}
