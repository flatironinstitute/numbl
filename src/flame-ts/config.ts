// BLIS/FLAME block size parameters.
// Tuned for typical desktop CPUs (L1=32KB, L2=256KB, L3=shared).
//
// mc×kc × 8 bytes should fit in L2
// kc×nc × 8 bytes should fit in L3
// mr×nr doubles should fit in JS engine registers

export const FLAME_CONFIG = {
  MC: 64,
  KC: 256,
  NC: 512,
  MR: 4,
  NR: 4,
  NB: 64, // LAPACK block size for dpotrf, dgetrf, dgeqrf
};
