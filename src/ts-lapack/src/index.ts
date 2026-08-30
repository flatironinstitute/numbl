// ts-lapack: Pure TypeScript translation of LAPACK routines
//
// All matrices are stored in column-major (Fortran) order in Float64Array.
// Element A(i,j) (1-based) is stored at index (i-1) + (j-1)*lda.
//
// NOTE: Float64Array is used (not Float32Array) because these are double-
// precision routines (the "D" prefix in DGETRF means double precision).

// LAPACK SRC
export { dgetrf } from "./SRC/dgetrf.js";
export { dgetrf2 } from "./SRC/dgetrf2.js";
export { dgetri } from "./SRC/dgetri.js";
export { dtrtri } from "./SRC/dtrtri.js";
export { dtrti2 } from "./SRC/dtrti2.js";
export { dlaswp } from "./SRC/dlaswp.js";
export { dlamch } from "./SRC/dlamch.js";

// BLAS Level 3
export { dgemm } from "./BLAS/dgemm.js";
export { dtrsm } from "./BLAS/dtrsm.js";
export { dtrmm } from "./BLAS/dtrmm.js";

// BLAS Level 2
export { dgemv } from "./BLAS/dgemv.js";
export { dger } from "./BLAS/dger.js";
export { dtrmv } from "./BLAS/dtrmv.js";

// BLAS Level 1
export { idamax } from "./BLAS/idamax.js";
export { dscal } from "./BLAS/dscal.js";
export { dswap } from "./BLAS/dswap.js";
export { daxpy } from "./BLAS/daxpy.js";
export { dnrm2 } from "./BLAS/dnrm2.js";

// LAPACK SRC – QR
export { dgeqrf } from "./SRC/dgeqrf.js";
export { dgeqr2 } from "./SRC/dgeqr2.js";
export { dorgqr } from "./SRC/dorgqr.js";
export { dorg2r } from "./SRC/dorg2r.js";
export { dlarfg } from "./SRC/dlarfg.js";
export { dlarf1f } from "./SRC/dlarf1f.js";
export { dlapy2 } from "./SRC/dlapy2.js";

// BLAS Level 1 (additional)
export { dcopy } from "./BLAS/dcopy.js";
export { ddot } from "./BLAS/ddot.js";
export { drot } from "./BLAS/drot.js";

// LAPACK SRC – utilities
export { dlacpy } from "./SRC/dlacpy.js";
export { dlaset } from "./SRC/dlaset.js";
export { dlassq } from "./SRC/dlassq.js";
export { dlange } from "./SRC/dlange.js";
export { dlascl } from "./SRC/dlascl.js";
export { dlartg } from "./SRC/dlartg.js";
export { dladiv } from "./SRC/dladiv.js";
export { dlanv2 } from "./SRC/dlanv2.js";
export { dlaln2 } from "./SRC/dlaln2.js";

// LAPACK SRC – Householder
export { dlarf } from "./SRC/dlarf.js";
export { dlarft } from "./SRC/dlarft.js";
export { dlarfb } from "./SRC/dlarfb.js";
export { dlarfx } from "./SRC/dlarfx.js";

// LAPACK SRC – Hessenberg reduction
export { dgehd2 } from "./SRC/dgehd2.js";
export { dlahr2 } from "./SRC/dlahr2.js";
export { dgehrd } from "./SRC/dgehrd.js";
export { dorghr } from "./SRC/dorghr.js";

// LAPACK SRC – orthogonal multiplication
export { dorm2r } from "./SRC/dorm2r.js";
export { dormqr } from "./SRC/dormqr.js";
export { dormhr } from "./SRC/dormhr.js";

// LAPACK SRC – QR algorithm for eigenvalues
export { dlasy2 } from "./SRC/dlasy2.js";
export { dlaqr1 } from "./SRC/dlaqr1.js";
export { dlahqr } from "./SRC/dlahqr.js";
export { dlaqr5 } from "./SRC/dlaqr5.js";
export { dlaexc } from "./SRC/dlaexc.js";
export { dtrexc } from "./SRC/dtrexc.js";
export { dlaqr2, dlaqr4 } from "./SRC/dlaqr24.js";
export { dlaqr3, dlaqr0 } from "./SRC/dlaqr30.js";
export { dhseqr } from "./SRC/dhseqr.js";

// LAPACK SRC – balancing & eigenvectors
export { dgebal } from "./SRC/dgebal.js";
export { dgebak } from "./SRC/dgebak.js";
export { dtrevc3 } from "./SRC/dtrevc3.js";
export { dgeev } from "./SRC/dgeev.js";

// LAPACK SRC – SVD
export { dlas2 } from "./SRC/dlas2.js";
export { dlasv2 } from "./SRC/dlasv2.js";
export { dlasr } from "./SRC/dlasr.js";
export { dlasrt } from "./SRC/dlasrt.js";
export { dgelq2 } from "./SRC/dgelq2.js";
export { dorml2 } from "./SRC/dorml2.js";
export { dorgl2 } from "./SRC/dorgl2.js";
export { dgelqf } from "./SRC/dgelqf.js";
export { dorglq } from "./SRC/dorglq.js";
export { dormlq } from "./SRC/dormlq.js";
export { dlabrd } from "./SRC/dlabrd.js";
export { dgebd2 } from "./SRC/dgebd2.js";
export { dgebrd } from "./SRC/dgebrd.js";
export { dbdsqr } from "./SRC/dbdsqr.js";
export { dorgbr } from "./SRC/dorgbr.js";
export { dormbr } from "./SRC/dormbr.js";
export { dgesvd } from "./SRC/dgesvd.js";

// LAPACK SRC – symmetric eigenvalue problem
export { dsymv } from "./BLAS/dsymv.js";
export { dsyr2 } from "./BLAS/dsyr2.js";
export { dlae2 } from "./SRC/dlae2.js";
export { dlaev2 } from "./SRC/dlaev2.js";
export { dlanst } from "./SRC/dlanst.js";
export { dlansy } from "./SRC/dlansy.js";
export { dsytd2 } from "./SRC/dsytd2.js";
export { dorgtr } from "./SRC/dorgtr.js";
export { dsteqr, COMPZ_N, COMPZ_V, COMPZ_I } from "./SRC/dsteqr.js";
export { dsyev } from "./SRC/dsyev.js";

// Utilities
export { xerbla } from "./utils/xerbla.js";
export * from "./utils/constants.js";
export { ilaenv } from "./utils/ilaenv.js";
export { iladlr } from "./utils/iladlr.js";
export { iladlc } from "./utils/iladlc.js";
