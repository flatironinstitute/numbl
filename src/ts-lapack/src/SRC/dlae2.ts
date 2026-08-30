// Translated from SRC/dlae2.f
// DLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix
//    [  A   B  ]
//    [  B   C  ].
// On return, RT1 is the eigenvalue of larger absolute value, and RT2 is the
// eigenvalue of smaller absolute value.
//
// RT1 is accurate to a few ulps barring over/underflow.  RT2 may be inaccurate
// if there is massive cancellation in the determinant A*C-B*B; higher
// precision or a symmetric algorithm can be used to compute RT2 more
// accurately.

export function dlae2(
  a: number,
  b: number,
  c: number
): { rt1: number; rt2: number } {
  const ONE = 1.0;
  const TWO = 2.0;
  const HALF = 0.5;

  // Compute the eigenvalues
  const sm = a + c;
  const df = a - c;
  const adf = Math.abs(df);
  const tb = b + b;
  const ab = Math.abs(tb);

  let acmx: number, acmn: number;
  if (Math.abs(a) > Math.abs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }

  let rt: number;
  if (adf > ab) {
    rt = adf * Math.sqrt(ONE + (ab / adf) ** 2);
  } else if (adf < ab) {
    rt = ab * Math.sqrt(ONE + (adf / ab) ** 2);
  } else {
    // Includes case AB=ADF=0
    rt = ab * Math.sqrt(TWO);
  }

  let rt1: number, rt2: number;
  if (sm < 0.0) {
    rt1 = HALF * (sm - rt);
    // Order of execution important. To get fully accurate smaller eigenvalue,
    // next line needs to be executed in higher precision.
    rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
  } else if (sm > 0.0) {
    rt1 = HALF * (sm + rt);
    rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
  } else {
    // Includes case RT1 = RT2 = 0
    rt1 = HALF * rt;
    rt2 = -HALF * rt;
  }

  return { rt1, rt2 };
}
