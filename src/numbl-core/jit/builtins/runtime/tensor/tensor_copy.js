// JS sibling of `tensor_copy.h`. Deep-copy a real tensor (e.g. `real(A)`
// or `conj(A)` of a real tensor), so the result can be mutated
// independently of the source.

export function jit_tensor_copy(t) {
  return {
    jitTag: "tensor",
    shape: t.shape.slice(),
    data: new Float64Array(t.data),
  };
}
