// Translated from SRC/dlaev2.f
// DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
//    [  A   B  ]
//    [  B   C  ].
// On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
// eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
// eigenvector for RT1, giving the decomposition
//
//    [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
//    [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].

export function dlaev2(
  a: number,
  b: number,
  c: number
): { rt1: number; rt2: number; cs1: number; sn1: number } {
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

  let rt1: number, rt2: number, sgn1: number;
  if (sm < 0.0) {
    rt1 = HALF * (sm - rt);
    sgn1 = -1;
    // Order of execution important. To get fully accurate smaller eigenvalue,
    // next line needs to be executed in higher precision.
    rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
  } else if (sm > 0.0) {
    rt1 = HALF * (sm + rt);
    sgn1 = 1;
    rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
  } else {
    // Includes case RT1 = RT2 = 0
    rt1 = HALF * rt;
    rt2 = -HALF * rt;
    sgn1 = 1;
  }

  // Compute the eigenvector
  let cs: number, sgn2: number;
  if (df >= 0.0) {
    cs = df + rt;
    sgn2 = 1;
  } else {
    cs = df - rt;
    sgn2 = -1;
  }

  let cs1: number, sn1: number;
  const acs = Math.abs(cs);
  if (acs > ab) {
    const ct = -tb / cs;
    sn1 = ONE / Math.sqrt(ONE + ct * ct);
    cs1 = ct * sn1;
  } else {
    if (ab === 0.0) {
      cs1 = ONE;
      sn1 = 0.0;
    } else {
      const tn = -cs / tb;
      cs1 = ONE / Math.sqrt(ONE + tn * tn);
      sn1 = tn * cs1;
    }
  }
  if (sgn1 === sgn2) {
    const tn = cs1;
    cs1 = -sn1;
    sn1 = tn;
  }

  return { rt1, rt2, cs1, sn1 };
}
