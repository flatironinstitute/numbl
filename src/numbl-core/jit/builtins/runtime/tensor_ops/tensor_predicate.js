// JS sibling of `tensor_predicate.h`. Real-tensor → logical-tensor
// predicate kernels for the js-aot backend, plus their `_complex`
// siblings (each reads `imag[i]` when defined, treats it as 0
// otherwise). Result carries `isLogical: true` so downstream index-
// slot resolution treats it as a mask.

function pred_kernel(a, fn) {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = fn(a.data[i]) ? 1 : 0;
  return {
    mtoc2Tag: "tensor",
    shape: a.shape.slice(),
    data: out,
    isLogical: true,
  };
}

function pred_kernel_complex(a, fn) {
  const out = new Float64Array(a.data.length);
  const im = a.imag;
  for (let i = 0; i < a.data.length; i++) {
    out[i] = fn(a.data[i], im !== undefined ? im[i] : 0) ? 1 : 0;
  }
  return {
    mtoc2Tag: "tensor",
    shape: a.shape.slice(),
    data: out,
    isLogical: true,
  };
}

export function mtoc2_tensor_isnan(a) {
  return pred_kernel(a, Number.isNaN);
}

export function mtoc2_tensor_logical(a) {
  return pred_kernel(a, x => x !== 0);
}

export function mtoc2_tensor_isinf(a) {
  return pred_kernel(a, x => x === Infinity || x === -Infinity);
}

export function mtoc2_tensor_isfinite(a) {
  return pred_kernel(a, Number.isFinite);
}

export function mtoc2_tensor_isnan_complex(a) {
  return pred_kernel_complex(
    a,
    (re, im) => Number.isNaN(re) || Number.isNaN(im)
  );
}

export function mtoc2_tensor_isinf_complex(a) {
  const isInf = x => x === Infinity || x === -Infinity;
  return pred_kernel_complex(a, (re, im) => isInf(re) || isInf(im));
}

export function mtoc2_tensor_isfinite_complex(a) {
  return pred_kernel_complex(
    a,
    (re, im) => Number.isFinite(re) && Number.isFinite(im)
  );
}
