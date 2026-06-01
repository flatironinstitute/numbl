// JS sibling of `tensor_imag_all_zero.h`. True when a tensor carries no
// imaginary content (no imag lane, or every imag element exactly zero).
// `isreal` uses this for complex-typed tensors the JIT could not prove
// real at compile time, so it reports realness by value — matching the
// interpreter and the complex-scalar `v.im === 0` rule.
export function mtoc2_tensor_imag_all_zero(a) {
  if (a.imag === undefined) return true;
  for (let i = 0; i < a.imag.length; i++) {
    if (a.imag[i] !== 0) return false;
  }
  return true;
}
