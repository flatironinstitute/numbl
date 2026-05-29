// JS sibling of `tensor_norm.h`. Vector norms for real and complex
// tensors. The complex variants tolerate a missing imag lane (real
// tensor flowing through a complex-typed norm route) by treating it
// as zero.

export function mtoc2_norm2_real(a) {
  let acc = 0;
  for (let i = 0; i < a.data.length; i++) {
    const x = a.data[i];
    acc += x * x;
  }
  return Math.sqrt(acc);
}

export function mtoc2_norm2_complex(a) {
  let acc = 0;
  const im = a.imag;
  for (let i = 0; i < a.data.length; i++) {
    const re = a.data[i];
    const imv = im !== undefined ? im[i] : 0;
    acc += re * re + imv * imv;
  }
  return Math.sqrt(acc);
}

export function mtoc2_norm_p_real(a, p) {
  const n = a.data.length;
  if (n === 0) return 0;
  if (p === Infinity) {
    let m = 0;
    for (let i = 0; i < n; i++) {
      const x = Math.abs(a.data[i]);
      if (x > m) m = x;
    }
    return m;
  }
  if (p === -Infinity) {
    let m = Math.abs(a.data[0]);
    for (let i = 1; i < n; i++) {
      const x = Math.abs(a.data[i]);
      if (x < m) m = x;
    }
    return m;
  }
  if (p === 1) {
    let acc = 0;
    for (let i = 0; i < n; i++) acc += Math.abs(a.data[i]);
    return acc;
  }
  if (p === 2) return mtoc2_norm2_real(a);
  let acc = 0;
  for (let i = 0; i < n; i++) acc += Math.pow(Math.abs(a.data[i]), p);
  return Math.pow(acc, 1 / p);
}

export function mtoc2_norm_p_complex(a, p) {
  const n = a.data.length;
  if (n === 0) return 0;
  const im = a.imag;
  const abs = i => {
    const re = a.data[i];
    const imv = im !== undefined ? im[i] : 0;
    return Math.hypot(re, imv);
  };
  if (p === Infinity) {
    let m = 0;
    for (let i = 0; i < n; i++) {
      const x = abs(i);
      if (x > m) m = x;
    }
    return m;
  }
  if (p === -Infinity) {
    let m = abs(0);
    for (let i = 1; i < n; i++) {
      const x = abs(i);
      if (x < m) m = x;
    }
    return m;
  }
  if (p === 2) return mtoc2_norm2_complex(a);
  if (p === 1) {
    let acc = 0;
    for (let i = 0; i < n; i++) acc += abs(i);
    return acc;
  }
  let acc = 0;
  for (let i = 0; i < n; i++) acc += Math.pow(abs(i), p);
  return Math.pow(acc, 1 / p);
}
