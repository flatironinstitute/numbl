const PINF = Number.POSITIVE_INFINITY;
const NINF = Number.NEGATIVE_INFINITY;

// -------------------------
// bit helper used by erf/erfc
// -------------------------
function setLowWord(x: number, low: number): number {
  const buf = new ArrayBuffer(8);
  const view = new DataView(buf);
  view.setFloat64(0, x, false); // big-endian
  view.setUint32(4, low >>> 0, false);
  return view.getFloat64(0, false);
}

// -------------------------
// erf / erfc polynomial helpers
// -------------------------
function polyvalPP(x: number): number {
  return (
    -0.3250421072470015 +
    x *
      (-0.02848174957559851 +
        x * (-0.005770270296489442 + x * -0.000023763016656650163))
  );
}

function polyvalQQ(x: number): number {
  return (
    0.39791722395915535 +
    x *
      (0.0650222499887673 +
        x *
          (0.005081306281875766 +
            x * (0.00013249473800432164 + x * -0.000003960228278775368)))
  );
}

function polyvalPA(x: number): number {
  return (
    0.41485611868374833 +
    x *
      (-0.3722078760357013 +
        x *
          (0.31834661990116175 +
            x *
              (-0.11089469428239668 +
                x * (0.035478304325618236 + x * -0.002166375594868791))))
  );
}

function polyvalQA(x: number): number {
  return (
    0.10642088040084423 +
    x *
      (0.540397917702171 +
        x *
          (0.07182865441419627 +
            x *
              (0.12617121980876164 +
                x * (0.01363708391202905 + x * 0.011984499846799107))))
  );
}

function polyvalRA(x: number): number {
  return (
    -0.6938585727071818 +
    x *
      (-10.558626225323291 +
        x *
          (-62.375332450326006 +
            x *
              (-162.39666946257347 +
                x *
                  (-184.60509290671104 +
                    x * (-81.2874355063066 + x * -9.814329344169145)))))
  );
}

function polyvalSA(x: number): number {
  return (
    19.651271667439257 +
    x *
      (137.65775414351904 +
        x *
          (434.56587747522923 +
            x *
              (645.3872717332679 +
                x *
                  (429.00814002756783 +
                    x *
                      (108.63500554177944 +
                        x * (6.570249770319282 + x * -0.0604244152148581))))))
  );
}

function polyvalRB(x: number): number {
  return (
    -0.799283237680523 +
    x *
      (-17.757954917754752 +
        x *
          (-160.63638485582192 +
            x *
              (-637.5664433683896 +
                x * (-1025.0951316110772 + x * -483.5191916086514))))
  );
}

function polyvalSB(x: number): number {
  return (
    30.33806074348246 +
    x *
      (325.7925129965739 +
        x *
          (1536.729586084437 +
            x *
              (3199.8582195085955 +
                x *
                  (2553.0504064331644 +
                    x * (474.52854120695537 + x * -22.44095244658582)))))
  );
}

// -------------------------
// erfinv / erfcinv rational helpers
// -------------------------
function rational_p1q1(x: number): number {
  const ax = Math.abs(x);
  let s1: number;
  let s2: number;
  if (x === 0.0) return -0.0005087819496582806;

  if (ax <= 1.0) {
    s1 =
      -0.0005087819496582806 +
      x *
        (-0.008368748197417368 +
          x *
            (0.03348066254097446 +
              x *
                (-0.012692614766297404 +
                  x *
                    (-0.03656379714117627 +
                      x *
                        (0.02198786811111689 +
                          x *
                            (0.008226878746769157 +
                              x * -0.005387729650712429))))));
    s2 =
      1.0 +
      x *
        (-0.9700050433032906 +
          x *
            (-1.5657455823417585 +
              x *
                (1.5622155839842302 +
                  x *
                    (0.662328840472003 +
                      x *
                        (-0.7122890234154284 +
                          x *
                            (-0.05273963823400997 +
                              x *
                                (0.07952836873415717 +
                                  x *
                                    (-0.0023339375937419 +
                                      x * 0.0008862163904564247))))))));
  } else {
    const y = 1.0 / x;
    s1 =
      y *
      (y *
        (-0.005387729650712429 +
          y *
            (0.008226878746769157 +
              y *
                (0.02198786811111689 +
                  y *
                    (-0.03656379714117627 +
                      y *
                        (-0.012692614766297404 +
                          y *
                            (0.03348066254097446 +
                              y *
                                (-0.008368748197417368 +
                                  y * -0.0005087819496582806))))))));
    s2 =
      0.0008862163904564247 +
      y *
        (-0.0023339375937419 +
          y *
            (0.07952836873415717 +
              y *
                (-0.05273963823400997 +
                  y *
                    (-0.7122890234154284 +
                      y *
                        (0.662328840472003 +
                          y *
                            (1.5622155839842302 +
                              y *
                                (-1.5657455823417585 +
                                  y * (-0.9700050433032906 + y))))))));
  }
  return s1 / s2;
}

function rational_p2q2(x: number): number {
  const ax = Math.abs(x);
  let s1: number;
  let s2: number;
  if (x === 0.0) return -0.20243350835593876;

  if (ax <= 1.0) {
    s1 =
      -0.20243350835593876 +
      x *
        (0.10526468069939171 +
          x *
            (8.3705032834312 +
              x *
                (17.644729840837403 +
                  x *
                    (-18.851064805871424 +
                      x *
                        (-44.6382324441787 +
                          x *
                            (17.445385985570866 +
                              x *
                                (21.12946554483405 +
                                  x * -3.6719225470772936)))))));
    s2 =
      1.0 +
      x *
        (6.242641248542475 +
          x *
            (3.971343795334387 +
              x *
                (-28.66081804998 +
                  x *
                    (-20.14326346804852 +
                      x *
                        (48.560921310873994 +
                          x *
                            (10.826866735546016 +
                              x *
                                (-22.643693341313973 +
                                  x * 1.7211476576120028)))))));
  } else {
    const y = 1.0 / x;
    s1 =
      -3.6719225470772936 +
      y *
        (21.12946554483405 +
          y *
            (17.445385985570866 +
              y *
                (-44.6382324441787 +
                  y *
                    (-18.851064805871424 +
                      y *
                        (17.644729840837403 +
                          y *
                            (8.3705032834312 +
                              y *
                                (0.10526468069939171 +
                                  y * -0.20243350835593876)))))));
    s2 =
      1.7211476576120028 +
      y *
        (-22.643693341313973 +
          y *
            (10.826866735546016 +
              y *
                (48.560921310873994 +
                  y *
                    (-20.14326346804852 +
                      y *
                        (-28.66081804998 +
                          y *
                            (3.971343795334387 +
                              y * (6.242641248542475 + y)))))));
  }
  return s1 / s2;
}

function rational_p3q3(x: number): number {
  const ax = Math.abs(x);
  let s1: number;
  let s2: number;
  if (x === 0.0) return -0.1311027816799519;

  if (ax <= 1.0) {
    s1 =
      -0.1311027816799519 +
      x *
        (-0.16379404719331705 +
          x *
            (0.11703015634199525 +
              x *
                (0.38707973897260434 +
                  x *
                    (0.3377855389120359 +
                      x *
                        (0.14286953440815717 +
                          x *
                            (0.029015791000532906 +
                              x *
                                (0.0021455899538880526 +
                                  x *
                                    (-6.794655751811263e-7 +
                                      x *
                                        (2.8522533178221704e-8 +
                                          x * -6.81149956853777e-10)))))))));
    s2 =
      1.0 +
      x *
        (3.4662540724256723 +
          x *
            (5.381683457070069 +
              x *
                (4.778465929458438 +
                  x *
                    (2.5930192162362027 +
                      x *
                        (0.848854343457902 +
                          x *
                            (0.15226433829533179 +
                              x * 0.011059242293464892))))));
  } else {
    const y = 1.0 / x;
    s1 =
      -6.81149956853777e-10 +
      y *
        (2.8522533178221704e-8 +
          y *
            (-6.794655751811263e-7 +
              y *
                (0.0021455899538880526 +
                  y *
                    (0.029015791000532906 +
                      y *
                        (0.14286953440815717 +
                          y *
                            (0.3377855389120359 +
                              y *
                                (0.38707973897260434 +
                                  y *
                                    (0.11703015634199525 +
                                      y *
                                        (-0.16379404719331705 +
                                          y * -0.1311027816799519)))))))));
    s2 =
      y *
      (y *
        (y *
          (0.011059242293464892 +
            y *
              (0.15226433829533179 +
                y *
                  (0.848854343457902 +
                    y *
                      (2.5930192162362027 +
                        y *
                          (4.778465929458438 +
                            y *
                              (5.381683457070069 +
                                y * (3.4662540724256723 + y)))))))));
  }
  return s1 / s2;
}

function rational_p4q4(x: number): number {
  const ax = Math.abs(x);
  let s1: number;
  let s2: number;
  if (x === 0.0) return -0.0350353787183178;

  if (ax <= 1.0) {
    s1 =
      -0.0350353787183178 +
      x *
        (-0.0022242652921344794 +
          x *
            (0.018557330651423107 +
              x *
                (0.009508047013259196 +
                  x *
                    (0.0018712349281955923 +
                      x *
                        (0.00015754461742496055 +
                          x *
                            (0.00000460469890584318 +
                              x *
                                (-2.304047769118826e-10 +
                                  x * 2.6633922742578204e-12)))))));
    s2 =
      1.0 +
      x *
        (1.3653349817554064 +
          x *
            (0.7620591645536234 +
              x *
                (0.22009110576413124 +
                  x *
                    (0.03415891436709477 +
                      x *
                        (0.00263861676657016 + x * 0.00007646752923027944)))));
  } else {
    const y = 1.0 / x;
    s1 =
      2.6633922742578204e-12 +
      y *
        (-2.304047769118826e-10 +
          y *
            (0.00000460469890584318 +
              y *
                (0.00015754461742496055 +
                  y *
                    (0.0018712349281955923 +
                      y *
                        (0.009508047013259196 +
                          y *
                            (0.018557330651423107 +
                              y *
                                (-0.0022242652921344794 +
                                  y * -0.0350353787183178)))))));
    s2 =
      y *
      (y *
        (0.00007646752923027944 +
          y *
            (0.00263861676657016 +
              y *
                (0.03415891436709477 +
                  y *
                    (0.22009110576413124 +
                      y *
                        (0.7620591645536234 +
                          y * (1.3653349817554064 + y)))))));
  }
  return s1 / s2;
}

function rational_p5q5(x: number): number {
  const ax = Math.abs(x);
  let s1: number;
  let s2: number;
  if (x === 0.0) return -0.016743100507663373;

  if (ax <= 1.0) {
    s1 =
      -0.016743100507663373 +
      x *
        (-0.0011295143874558028 +
          x *
            (0.001056288621524929 +
              x *
                (0.00020938631748758808 +
                  x *
                    (0.000014962478375834237 +
                      x *
                        (4.4969678992770644e-7 +
                          x *
                            (4.625961635228786e-9 +
                              x *
                                (-2.811287356288318e-14 +
                                  x * 9.905570997331033e-17)))))));
    s2 =
      1.0 +
      x *
        (0.5914293448864175 +
          x *
            (0.1381518657490833 +
              x *
                (0.016074608709367652 +
                  x *
                    (0.0009640118070051656 +
                      x *
                        (0.000027533547476472603 + x * 2.82243172016108e-7)))));
  } else {
    const y = 1.0 / x;
    s1 =
      9.905570997331033e-17 +
      y *
        (-2.811287356288318e-14 +
          y *
            (4.625961635228786e-9 +
              y *
                (4.4969678992770644e-7 +
                  y *
                    (0.000014962478375834237 +
                      y *
                        (0.00020938631748758808 +
                          y *
                            (0.001056288621524929 +
                              y *
                                (-0.0011295143874558028 +
                                  y * -0.016743100507663373)))))));
    s2 =
      y *
      (y *
        (2.82243172016108e-7 +
          y *
            (0.000027533547476472603 +
              y *
                (0.0009640118070051656 +
                  y *
                    (0.016074608709367652 +
                      y *
                        (0.1381518657490833 +
                          y * (0.5914293448864175 + y)))))));
  }
  return s1 / s2;
}

// -------------------------
// erf
// Source lineage: Sun/FreeBSD via stdlib-js
// -------------------------
export const erfScalar = (x: number): number => {
  const TINY = 1.0e-300;
  const VERY_TINY = 2.848094538889218e-306;
  const SMALL = 3.725290298461914e-9;
  const ERX = 8.45062911510467529297e-1;
  const EFX = 1.28379167095512586316e-1;
  const EFX8 = 1.02703333676410069053;
  const PPC = 1.28379167095512558561e-1;
  const QQC = 1.0;
  const PAC = -2.36211856075265944077e-3;
  const QAC = 1.0;
  const RAC = -9.86494403484714822705e-3;
  const SAC = 1.0;
  const RBC = -9.86494292470009928597e-3;
  const SBC = 1.0;

  let sign: boolean;
  let ax: number;
  let z: number;
  let r: number;
  let s: number;
  let y: number;
  let p: number;
  let q: number;

  if (Number.isNaN(x)) return NaN;
  if (x === PINF) return 1.0;
  if (x === NINF) return -1.0;
  if (x === 0.0) return x;

  if (x < 0.0) {
    sign = true;
    ax = -x;
  } else {
    sign = false;
    ax = x;
  }

  if (ax < 0.84375) {
    if (ax < SMALL) {
      if (ax < VERY_TINY) {
        return 0.125 * (8.0 * x + EFX8 * x);
      }
      return x + EFX * x;
    }
    z = x * x;
    r = PPC + z * polyvalPP(z);
    s = QQC + z * polyvalQQ(z);
    y = r / s;
    return x + x * y;
  }

  if (ax < 1.25) {
    s = ax - 1.0;
    p = PAC + s * polyvalPA(s);
    q = QAC + s * polyvalQA(s);
    return sign ? -ERX - p / q : ERX + p / q;
  }

  if (ax >= 6.0) {
    return sign ? TINY - 1.0 : 1.0 - TINY;
  }

  s = 1.0 / (ax * ax);

  if (ax < 2.857142857142857) {
    r = RAC + s * polyvalRA(s);
    s = SAC + s * polyvalSA(s);
  } else {
    r = RBC + s * polyvalRB(s);
    s = SBC + s * polyvalSB(s);
  }

  z = setLowWord(ax, 0);
  r = Math.exp(-(z * z) - 0.5625) * Math.exp((z - ax) * (z + ax) + r / s);
  return sign ? r / ax - 1.0 : 1.0 - r / ax;
};

// -------------------------
// erfc
// Source lineage: Sun/FreeBSD via stdlib-js
// -------------------------
export const erfcScalar = (x: number): number => {
  const TINY = 1.0e-300;
  const SMALL = 1.3877787807814457e-17;
  const ERX = 8.45062911510467529297e-1;
  const PPC = 1.28379167095512558561e-1;
  const QQC = 1.0;
  const PAC = -2.36211856075265944077e-3;
  const QAC = 1.0;
  const RAC = -9.86494403484714822705e-3;
  const SAC = 1.0;
  const RBC = -9.86494292470009928597e-3;
  const SBC = 1.0;

  let sign: boolean;
  let ax: number;
  let z: number;
  let r: number;
  let s: number;
  let y: number;
  let p: number;
  let q: number;

  if (Number.isNaN(x)) return NaN;
  if (x === PINF) return 0.0;
  if (x === NINF) return 2.0;
  if (x === 0.0) return 1.0;

  if (x < 0.0) {
    sign = true;
    ax = -x;
  } else {
    sign = false;
    ax = x;
  }

  if (ax < 0.84375) {
    if (ax < SMALL) return 1.0 - x;

    z = x * x;
    r = PPC + z * polyvalPP(z);
    s = QQC + z * polyvalQQ(z);
    y = r / s;

    if (x < 0.25) {
      return 1.0 - (x + x * y);
    }
    r = x * y;
    r += x - 0.5;
    return 0.5 - r;
  }

  if (ax < 1.25) {
    s = ax - 1.0;
    p = PAC + s * polyvalPA(s);
    q = QAC + s * polyvalQA(s);
    return sign ? 1.0 + ERX + p / q : 1.0 - ERX - p / q;
  }

  if (ax < 28.0) {
    s = 1.0 / (ax * ax);

    if (ax < 2.857142857142857) {
      r = RAC + s * polyvalRA(s);
      s = SAC + s * polyvalSA(s);
    } else {
      if (x < -6.0) return 2.0 - TINY;
      r = RBC + s * polyvalRB(s);
      s = SBC + s * polyvalSB(s);
    }

    z = setLowWord(ax, 0);
    r = Math.exp(-(z * z) - 0.5625) * Math.exp((z - ax) * (z + ax) + r / s);
    return sign ? 2.0 - r / ax : r / ax;
  }

  return sign ? 2.0 - TINY : TINY * TINY;
};

// -------------------------
// erfinv
// Source lineage: Boost via stdlib-js
// -------------------------
export const erfinvScalar = (x: number): number => {
  const Y1 = 8.91314744949340820313e-2;
  const Y2 = 2.249481201171875;
  const Y3 = 8.07220458984375e-1;
  const Y4 = 9.3995571136474609375e-1;
  const Y5 = 9.8362827301025390625e-1;

  let sign: number;
  let ax: number;
  let qs: number;
  let q: number;
  let g: number;
  let r: number;

  if (Number.isNaN(x)) return NaN;
  if (x === 1.0) return PINF;
  if (x === -1.0) return NINF;
  if (x === 0.0) return x;
  if (x > 1.0 || x < -1.0) return NaN;

  if (x < 0.0) {
    sign = -1.0;
    ax = -x;
  } else {
    sign = 1.0;
    ax = x;
  }

  q = 1.0 - ax;

  if (ax <= 0.5) {
    g = ax * (ax + 10.0);
    r = rational_p1q1(ax);
    return sign * (g * Y1 + g * r);
  }

  if (q >= 0.25) {
    g = Math.sqrt(-2.0 * Math.log(q));
    q -= 0.25;
    r = rational_p2q2(q);
    return sign * (g / (Y2 + r));
  }

  q = Math.sqrt(-Math.log(q));

  if (q < 3.0) {
    qs = q - 1.125;
    r = rational_p3q3(qs);
    return sign * (Y3 * q + r * q);
  }

  if (q < 6.0) {
    qs = q - 3.0;
    r = rational_p4q4(qs);
    return sign * (Y4 * q + r * q);
  }

  qs = q - 6.0;
  r = rational_p5q5(qs);
  return sign * (Y5 * q + r * q);
};

// -------------------------
// erfcinv
// Source lineage: Boost via stdlib-js
// -------------------------
export const erfcinvScalar = (x: number): number => {
  const Y1 = 8.91314744949340820313e-2;
  const Y2 = 2.249481201171875;
  const Y3 = 8.07220458984375e-1;
  const Y4 = 9.3995571136474609375e-1;
  const Y5 = 9.8362827301025390625e-1;

  let sign: number;
  let qs: number;
  let q: number;
  let g: number;
  let r: number;

  if (Number.isNaN(x)) return NaN;
  if (x === 0.0) return PINF;
  if (x === 2.0) return NINF;
  if (x === 1.0) return 0.0;
  if (x > 2.0 || x < 0.0) return NaN;

  if (x > 1.0) {
    sign = -1.0;
    q = 2.0 - x;
  } else {
    sign = 1.0;
    q = x;
  }

  x = 1.0 - q;

  if (x <= 0.5) {
    g = x * (x + 10.0);
    r = rational_p1q1(x);
    return sign * (g * Y1 + g * r);
  }

  if (q >= 0.25) {
    g = Math.sqrt(-2.0 * Math.log(q));
    q -= 0.25;
    r = rational_p2q2(q);
    return sign * (g / (Y2 + r));
  }

  q = Math.sqrt(-Math.log(q));

  if (q < 3.0) {
    qs = q - 1.125;
    r = rational_p3q3(qs);
    return sign * (Y3 * q + r * q);
  }

  if (q < 6.0) {
    qs = q - 3.0;
    r = rational_p4q4(qs);
    return sign * (Y4 * q + r * q);
  }

  qs = q - 6.0;
  r = rational_p5q5(qs);
  return sign * (Y5 * q + r * q);
};

// -------------------------
// erfcx  –  scaled complementary error function
// erfcx(x) = exp(x²) * erfc(x)
// -------------------------
export const erfcxScalar = (x: number): number => {
  if (Number.isNaN(x)) return NaN;
  if (x === PINF) return 0.0;
  if (x === NINF) return PINF;

  const ax = Math.abs(x);

  if (ax < 28) {
    // erfc(x) won't underflow for |x| < 28; compute directly
    return Math.exp(x * x) * erfcScalar(x);
  }

  // Large positive x (>= 28): erfc underflows, use asymptotic expansion
  // erfcx(x) = 1/(x*sqrt(pi)) * (1 - 1/(2x²) + 3/(4x⁴) - ...)
  if (x > 0) {
    const SQRT_PI = 1.7724538509055159;
    const x2inv = 1.0 / (x * x);
    return (1.0 / (x * SQRT_PI)) * (1.0 - 0.5 * x2inv + 0.75 * x2inv * x2inv);
  }

  // Large negative x (< -28): erfcx(x) = exp(x²)*erfc(x) → ∞
  return PINF;
};
