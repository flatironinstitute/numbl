/**
 * gmres() — Restarted GMRES solver using BLAS for matvec and LAPACK for
 * preconditioner solves.
 *
 *   gmres(A, n, b, restart, tol, maxit, M1, M2, x0):
 *     { x: Float64Array, flag: number, relres: number,
 *       iter: Int32Array, resvec: Float64Array }
 *
 *   A  : Float64Array (n×n column-major)
 *   n  : number
 *   b  : Float64Array (n)
 *   restart : number  (inner iterations per cycle)
 *   tol     : number
 *   maxit   : number  (max outer iterations)
 *   M1 : Float64Array (n×n) | null   — preconditioner factor 1
 *   M2 : Float64Array (n×n) | null   — preconditioner factor 2
 *   x0 : Float64Array (n)   | null   — initial guess
 */

#include "numbl_addon_common.h"
#include <cmath>

// ── Local helpers ────────────────────────────────────────────────────────────

static inline double vec_dot(int n, const double* a, const double* b) {
  double s = 0;
  for (int i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

static inline double vec_nrm2(int n, const double* a) {
  double s = 0;
  for (int i = 0; i < n; i++) s += a[i] * a[i];
  return std::sqrt(s);
}

// y = A*x  via BLAS dgemv
static inline void mat_vec(int n, const double* A, const double* x, double* y) {
  char trans = 'N';
  double alpha = 1.0, beta = 0.0;
  int inc = 1;
  dgemv_(&trans, &n, &n, &alpha, const_cast<double*>(A), &n,
         const_cast<double*>(x), &inc, &beta, y, &inc);
}

// Solve using pre-factored LU  (dgetrs, in-place on rhs)
static inline void lu_solve(int n, const double* LU, const int* ipiv, double* rhs) {
  char trans = 'N';
  int nrhs = 1, info_val = 0;
  dgetrs_(&trans, &n, &nrhs, const_cast<double*>(LU), &n,
          const_cast<int*>(ipiv), rhs, &n, &info_val);
}

static inline void givens(double a, double b, double& c, double& s) {
  if (b == 0.0) { c = 1.0; s = 0.0; return; }
  if (std::abs(b) > std::abs(a)) {
    double t = a / b;
    s = 1.0 / std::sqrt(1.0 + t * t);
    c = s * t;
  } else {
    double t = b / a;
    c = 1.0 / std::sqrt(1.0 + t * t);
    s = c * t;
  }
}

// ── Gmres ────────────────────────────────────────────────────────────────────

Napi::Value Gmres(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 9) {
    Napi::TypeError::New(env, "gmres: expected 9 arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // ── Parse required args ────────────────────────────────────────────────────
  if (!info[0].IsTypedArray() || !info[1].IsNumber() || !info[2].IsTypedArray()
      || !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsNumber()) {
    Napi::TypeError::New(env,
        "gmres: expected (Float64Array A, number n, Float64Array b, "
        "number restart, number tol, number maxit, M1?, M2?, x0?)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto fA = info[0].As<Napi::Float64Array>();
  int n = info[1].As<Napi::Number>().Int32Value();
  auto fB = info[2].As<Napi::Float64Array>();

  if (n <= 0 || static_cast<int>(fA.ElementLength()) != n * n
             || static_cast<int>(fB.ElementLength()) != n) {
    Napi::RangeError::New(env, "gmres: dimension mismatch")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int restart = info[3].As<Napi::Number>().Int32Value();
  double tol  = info[4].As<Napi::Number>().DoubleValue();
  int maxit   = info[5].As<Napi::Number>().Int32Value();

  if (restart <= 0 || restart > n) restart = n;

  const double* A = fA.Data();
  const double* b = fB.Data();

  // ── Pre-factor preconditioners ─────────────────────────────────────────────
  bool hasM1 = !info[6].IsNull() && !info[6].IsUndefined();
  bool hasM2 = !info[7].IsNull() && !info[7].IsUndefined();

  std::vector<double> m1lu, m2lu;
  std::vector<int> m1ipiv, m2ipiv;

  if (hasM1) {
    auto fM1 = info[6].As<Napi::Float64Array>();
    if (static_cast<int>(fM1.ElementLength()) != n * n) {
      Napi::RangeError::New(env, "gmres: M1 size mismatch")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    m1lu.assign(fM1.Data(), fM1.Data() + n * n);
    m1ipiv.resize(n);
    int info_val = 0;
    dgetrf_(&n, &n, m1lu.data(), &n, m1ipiv.data(), &info_val);
    if (info_val != 0) {
      Napi::Error::New(env, "gmres: preconditioner M1 is singular")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
  }
  if (hasM2) {
    auto fM2 = info[7].As<Napi::Float64Array>();
    if (static_cast<int>(fM2.ElementLength()) != n * n) {
      Napi::RangeError::New(env, "gmres: M2 size mismatch")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    m2lu.assign(fM2.Data(), fM2.Data() + n * n);
    m2ipiv.resize(n);
    int info_val = 0;
    dgetrf_(&n, &n, m2lu.data(), &n, m2ipiv.data(), &info_val);
    if (info_val != 0) {
      Napi::Error::New(env, "gmres: preconditioner M2 is singular")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  // ── Parse x0 ──────────────────────────────────────────────────────────────
  std::vector<double> x(n, 0.0);
  if (!info[8].IsNull() && !info[8].IsUndefined()) {
    auto fX0 = info[8].As<Napi::Float64Array>();
    if (static_cast<int>(fX0.ElementLength()) != n) {
      Napi::RangeError::New(env, "gmres: x0 size mismatch")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    std::memcpy(x.data(), fX0.Data(), n * sizeof(double));
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  auto apply_precond = [&](double* r) {
    if (hasM1) lu_solve(n, m1lu.data(), m1ipiv.data(), r);
    if (hasM2) lu_solve(n, m2lu.data(), m2ipiv.data(), r);
  };

  // ── Initial preconditioned residual ────────────────────────────────────────
  std::vector<double> r(n), tmp(n);
  mat_vec(n, A, x.data(), r.data());          // r = A*x
  for (int i = 0; i < n; i++) r[i] = b[i] - r[i]; // r = b - A*x
  apply_precond(r.data());                     // r = M\(b - A*x)
  double beta = vec_nrm2(n, r.data());

  // Preconditioned RHS norm
  std::vector<double> Mb(b, b + n);
  apply_precond(Mb.data());
  double normMb = vec_nrm2(n, Mb.data());
  if (normMb == 0.0) normMb = 1.0;

  std::vector<double> resvec;
  resvec.push_back(beta);

  int flag = 1;
  int outerIter = 0, innerIter = 0;

  // Already converged?
  if (beta / normMb <= tol) {
    flag = 0;
    goto done;
  }

  {
    // Workspace allocated once and reused across restarts
    std::vector<double> V(n * (restart + 1));
    std::vector<double> H((restart + 1) * restart);
    std::vector<double> cs(restart), sn(restart);
    std::vector<double> g(restart + 1);
    std::vector<double> w(n);

    for (int outer = 1; outer <= maxit; outer++) {
      outerIter = outer;

      // V(:,0) = r / beta
      for (int i = 0; i < n; i++) V[i] = r[i] / beta;

      std::fill(H.begin(), H.end(), 0.0);
      std::fill(cs.begin(), cs.end(), 0.0);
      std::fill(sn.begin(), sn.end(), 0.0);
      std::fill(g.begin(), g.end(), 0.0);
      g[0] = beta;

      bool converged = false;
      const int ldh = restart + 1;

      for (int j = 0; j < restart; j++) {
        innerIter = j + 1;

        // w = A * V(:,j)
        mat_vec(n, A, &V[j * n], w.data());
        apply_precond(w.data());

        // Modified Gram-Schmidt
        for (int i = 0; i <= j; i++) {
          double hij = vec_dot(n, w.data(), &V[i * n]);
          H[i + j * ldh] = hij;
          for (int k = 0; k < n; k++) w[k] -= hij * V[k + i * n];
        }

        double wnorm = vec_nrm2(n, w.data());
        H[(j + 1) + j * ldh] = wnorm;

        if (wnorm > 1e-300) {
          double inv_wnorm = 1.0 / wnorm;
          for (int k = 0; k < n; k++) V[k + (j + 1) * n] = w[k] * inv_wnorm;
        }

        // Apply previous Givens rotations
        for (int i = 0; i < j; i++) {
          double hi  = H[i + j * ldh];
          double hi1 = H[(i + 1) + j * ldh];
          H[i + j * ldh]       =  cs[i] * hi + sn[i] * hi1;
          H[(i + 1) + j * ldh] = -sn[i] * hi + cs[i] * hi1;
        }

        // New Givens rotation
        double c, s;
        givens(H[j + j * ldh], H[(j + 1) + j * ldh], c, s);
        cs[j] = c;
        sn[j] = s;

        H[j + j * ldh] = c * H[j + j * ldh] + s * H[(j + 1) + j * ldh];
        H[(j + 1) + j * ldh] = 0.0;

        double gj = g[j];
        g[j]     =  c * gj;
        g[j + 1] = -s * gj;

        double residNorm = std::abs(g[j + 1]);
        resvec.push_back(residNorm);

        if (residNorm / normMb <= tol) {
          // Back-solve H(0:j+1, 0:j+1) * y = g(0:j+1)
          std::vector<double> y(j + 1);
          for (int k = 0; k <= j; k++) y[k] = g[k];
          for (int k = j; k >= 0; k--) {
            for (int l = k + 1; l <= j; l++) y[k] -= H[k + l * ldh] * y[l];
            y[k] /= H[k + k * ldh];
          }
          // x += V(:,0:j+1) * y
          for (int k = 0; k < n; k++)
            for (int l = 0; l <= j; l++) x[k] += V[k + l * n] * y[l];

          flag = 0;
          converged = true;
          break;
        }
      }

      if (converged) break;

      // Restart: solve and update x
      {
        std::vector<double> y(restart);
        for (int k = 0; k < restart; k++) y[k] = g[k];
        for (int k = restart - 1; k >= 0; k--) {
          for (int l = k + 1; l < restart; l++) y[k] -= H[k + l * ldh] * y[l];
          y[k] /= H[k + k * ldh];
        }
        for (int k = 0; k < n; k++)
          for (int l = 0; l < restart; l++) x[k] += V[k + l * n] * y[l];
      }

      // Recompute residual
      mat_vec(n, A, x.data(), r.data());
      for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
      apply_precond(r.data());
      beta = vec_nrm2(n, r.data());

      if (beta / normMb <= tol) {
        flag = 0;
        innerIter = 0;
        break;
      }
    }
  }

done:
  // Final relative residual (recompute for accuracy if converged)
  double relres;
  if (flag == 0) {
    mat_vec(n, A, x.data(), tmp.data());
    for (int i = 0; i < n; i++) tmp[i] = b[i] - tmp[i];
    apply_precond(tmp.data());
    relres = vec_nrm2(n, tmp.data()) / normMb;
  } else {
    relres = beta / normMb;
  }

  auto result = Napi::Object::New(env);
  result.Set("x", vecToF64(env, x));
  result.Set("flag", Napi::Number::New(env, flag));
  result.Set("relres", Napi::Number::New(env, relres));
  auto iterArr = Napi::Int32Array::New(env, 2);
  iterArr[0] = outerIter;
  iterArr[1] = innerIter;
  result.Set("iter", iterArr);
  result.Set("resvec", vecToF64(env, resvec));
  return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// GmresComplex — complex version using BLAS zgemv + LAPACK zgetrf/zgetrs
// ══════════════════════════════════════════════════════════════════════════════

using cx = lapack_complex_double;

static inline double cx_abs(cx a) { return std::sqrt(a.real*a.real + a.imag*a.imag); }
static inline cx cx_conj(cx a) { return {a.real, -a.imag}; }
static inline cx cx_mul(cx a, cx b) { return {a.real*b.real - a.imag*b.imag, a.real*b.imag + a.imag*b.real}; }
static inline cx cx_add(cx a, cx b) { return {a.real+b.real, a.imag+b.imag}; }
static inline cx cx_sub(cx a, cx b) { return {a.real-b.real, a.imag-b.imag}; }
static inline cx cx_scale(double s, cx a) { return {s*a.real, s*a.imag}; }
static inline cx cx_div(cx a, cx b) {
  double d = b.real*b.real + b.imag*b.imag;
  return {(a.real*b.real + a.imag*b.imag)/d, (a.imag*b.real - a.real*b.imag)/d};
}

static double cvec_nrm2(int n, const cx* a) {
  double s = 0;
  for (int i = 0; i < n; i++) s += a[i].real*a[i].real + a[i].imag*a[i].imag;
  return std::sqrt(s);
}

// Conjugate dot product: sum(conj(a) * b)
static cx cvec_dot(int n, const cx* a, const cx* b) {
  double re = 0, im = 0;
  for (int i = 0; i < n; i++) {
    re += a[i].real*b[i].real + a[i].imag*b[i].imag;
    im += a[i].real*b[i].imag - a[i].imag*b[i].real;
  }
  return {re, im};
}

// y = A*x using zgemv
static void cmat_vec(int n, const cx* A, const cx* x, cx* y) {
  char trans = 'N';
  cx alpha = {1,0}, beta_val = {0,0};
  int inc = 1;
  zgemv_(&trans, &n, &n, &alpha, const_cast<cx*>(A), &n,
         const_cast<cx*>(x), &inc, &beta_val, y, &inc);
}

static void clu_solve(int n, const cx* LU, const int* ipiv, cx* rhs) {
  char trans = 'N';
  int nrhs = 1, info_val = 0;
  zgetrs_(&trans, &n, &nrhs, const_cast<cx*>(LU), &n,
          const_cast<int*>(ipiv), rhs, &n, &info_val);
}

// Complex Givens: [c s; -conj(s) c] * [a; b] = [r; 0], c real >= 0
static void cgivens(cx a, cx b, double& c, cx& s, cx& r) {
  double absB = cx_abs(b);
  if (absB == 0) { c = 1; s = {0,0}; r = a; return; }
  double absA = cx_abs(a);
  if (absA == 0) { c = 0; s = cx_conj({b.real/absB, b.imag/absB}); r = {absB,0}; return; }
  double norm = std::sqrt(absA*absA + absB*absB);
  c = absA / norm;
  cx alpha = {a.real/absA, a.imag/absA};
  s = cx_div(cx_mul(alpha, cx_conj(b)), {norm, 0});
  r = cx_scale(norm, alpha);
}

Napi::Value GmresComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // Args: ARe, AIm, n, bRe, bIm, restart, tol, maxit,
  //       M1Re, M1Im, M2Re, M2Im, x0Re, x0Im  (14 args)
  if (info.Length() < 14) {
    Napi::TypeError::New(env, "gmresComplex: expected 14 arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto fARe = info[0].As<Napi::Float64Array>();
  auto fAIm = info[1].As<Napi::Float64Array>();
  int n = info[2].As<Napi::Number>().Int32Value();
  auto fBRe = info[3].As<Napi::Float64Array>();
  auto fBIm = info[4].As<Napi::Float64Array>();
  int restart = info[5].As<Napi::Number>().Int32Value();
  double tol = info[6].As<Napi::Number>().DoubleValue();
  int maxit = info[7].As<Napi::Number>().Int32Value();

  if (restart <= 0 || restart > n) restart = n;

  // Interleave A
  auto A = splitToInterleaved(fARe, fAIm, n * n);

  // b
  std::vector<cx> b(n);
  for (int i = 0; i < n; i++) b[i] = {fBRe[i], fBIm[i]};

  // Pre-factor preconditioners
  bool hasM1 = !info[8].IsNull() && !info[8].IsUndefined();
  bool hasM2 = !info[10].IsNull() && !info[10].IsUndefined();

  std::vector<cx> m1lu, m2lu;
  std::vector<int> m1ipiv, m2ipiv;

  if (hasM1) {
    auto fM1Re = info[8].As<Napi::Float64Array>();
    auto fM1Im = info[9].As<Napi::Float64Array>();
    m1lu = splitToInterleaved(fM1Re, fM1Im, n * n);
    m1ipiv.resize(n);
    int info_val = 0;
    zgetrf_(&n, &n, m1lu.data(), &n, m1ipiv.data(), &info_val);
    if (info_val != 0) {
      Napi::Error::New(env, "gmresComplex: M1 is singular").ThrowAsJavaScriptException();
      return env.Null();
    }
  }
  if (hasM2) {
    auto fM2Re = info[10].As<Napi::Float64Array>();
    auto fM2Im = info[11].As<Napi::Float64Array>();
    m2lu = splitToInterleaved(fM2Re, fM2Im, n * n);
    m2ipiv.resize(n);
    int info_val = 0;
    zgetrf_(&n, &n, m2lu.data(), &n, m2ipiv.data(), &info_val);
    if (info_val != 0) {
      Napi::Error::New(env, "gmresComplex: M2 is singular").ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  // x0
  std::vector<cx> x(n, {0,0});
  if (!info[12].IsNull() && !info[12].IsUndefined()) {
    auto fX0Re = info[12].As<Napi::Float64Array>();
    auto fX0Im = info[13].As<Napi::Float64Array>();
    for (int i = 0; i < n; i++) x[i] = {fX0Re[i], fX0Im[i]};
  }

  auto apply_precond = [&](cx* r) {
    if (hasM1) clu_solve(n, m1lu.data(), m1ipiv.data(), r);
    if (hasM2) clu_solve(n, m2lu.data(), m2ipiv.data(), r);
  };

  // Initial residual
  std::vector<cx> r(n), tmp(n);
  cmat_vec(n, A.data(), x.data(), r.data());
  for (int i = 0; i < n; i++) r[i] = cx_sub(b[i], r[i]);
  apply_precond(r.data());
  double beta = cvec_nrm2(n, r.data());

  std::vector<cx> Mb(b);
  apply_precond(Mb.data());
  double normMb = cvec_nrm2(n, Mb.data());
  if (normMb == 0.0) normMb = 1.0;

  std::vector<double> resvec;
  resvec.push_back(beta);

  int flag = 1, outerIter = 0, innerIter = 0;

  if (beta / normMb <= tol) { flag = 0; goto cdone; }

  {
    std::vector<cx> V(n * (restart + 1));
    std::vector<cx> H((restart + 1) * restart, {0,0});
    std::vector<double> cs_arr(restart);
    std::vector<cx> sn_arr(restart);
    std::vector<cx> g(restart + 1, {0,0});
    std::vector<cx> w(n);

    for (int outer = 1; outer <= maxit; outer++) {
      outerIter = outer;
      for (int i = 0; i < n; i++) V[i] = cx_scale(1.0/beta, r[i]);

      std::fill(H.begin(), H.end(), cx{0,0});
      std::fill(cs_arr.begin(), cs_arr.end(), 0.0);
      std::fill(sn_arr.begin(), sn_arr.end(), cx{0,0});
      std::fill(g.begin(), g.end(), cx{0,0});
      g[0] = {beta, 0};

      bool converged = false;
      const int ldh = restart + 1;

      for (int j = 0; j < restart; j++) {
        innerIter = j + 1;

        cmat_vec(n, A.data(), &V[j*n], w.data());
        apply_precond(w.data());

        // Modified Gram-Schmidt (conjugate dot)
        for (int i = 0; i <= j; i++) {
          cx hij = cvec_dot(n, &V[i*n], w.data());
          H[i + j*ldh] = hij;
          for (int k = 0; k < n; k++) w[k] = cx_sub(w[k], cx_mul(hij, V[k + i*n]));
        }

        double wnorm = cvec_nrm2(n, w.data());
        H[(j+1) + j*ldh] = {wnorm, 0};
        if (wnorm > 1e-300) {
          double inv_w = 1.0 / wnorm;
          for (int k = 0; k < n; k++) V[k + (j+1)*n] = cx_scale(inv_w, w[k]);
        }

        // Apply previous Givens rotations
        for (int i = 0; i < j; i++) {
          double c = cs_arr[i]; cx s = sn_arr[i];
          cx hi = H[i + j*ldh], hi1 = H[(i+1) + j*ldh];
          H[i + j*ldh]     = cx_add(cx_scale(c, hi), cx_mul(s, hi1));
          H[(i+1) + j*ldh] = cx_add(cx_mul({-s.real, s.imag}, hi), cx_scale(c, hi1));
        }

        // New Givens rotation
        double c; cx s, rr;
        cgivens(H[j + j*ldh], H[(j+1) + j*ldh], c, s, rr);
        cs_arr[j] = c;
        sn_arr[j] = s;

        H[j + j*ldh] = rr;
        H[(j+1) + j*ldh] = {0,0};

        cx gj = g[j], gj1 = g[j+1];
        g[j]   = cx_add(cx_scale(c, gj), cx_mul(s, gj1));
        g[j+1] = cx_add(cx_mul({-s.real, s.imag}, gj), cx_scale(c, gj1));

        double residNorm = cx_abs(g[j+1]);
        resvec.push_back(residNorm);

        if (residNorm / normMb <= tol) {
          // Complex back-solve
          std::vector<cx> y(j+1);
          for (int k = 0; k <= j; k++) y[k] = g[k];
          for (int k = j; k >= 0; k--) {
            for (int l = k+1; l <= j; l++) y[k] = cx_sub(y[k], cx_mul(H[k + l*ldh], y[l]));
            y[k] = cx_div(y[k], H[k + k*ldh]);
          }
          for (int k = 0; k < n; k++)
            for (int l = 0; l <= j; l++) x[k] = cx_add(x[k], cx_mul(V[k + l*n], y[l]));
          flag = 0; converged = true; break;
        }
      }
      if (converged) break;

      // Restart
      {
        std::vector<cx> y(restart);
        for (int k = 0; k < restart; k++) y[k] = g[k];
        for (int k = restart-1; k >= 0; k--) {
          for (int l = k+1; l < restart; l++) y[k] = cx_sub(y[k], cx_mul(H[k + l*ldh], y[l]));
          y[k] = cx_div(y[k], H[k + k*ldh]);
        }
        for (int k = 0; k < n; k++)
          for (int l = 0; l < restart; l++) x[k] = cx_add(x[k], cx_mul(V[k + l*n], y[l]));
      }

      cmat_vec(n, A.data(), x.data(), r.data());
      for (int i = 0; i < n; i++) r[i] = cx_sub(b[i], r[i]);
      apply_precond(r.data());
      beta = cvec_nrm2(n, r.data());
      if (beta / normMb <= tol) { flag = 0; innerIter = 0; break; }
    }
  }

cdone:
  double relres;
  if (flag == 0) {
    cmat_vec(n, A.data(), x.data(), tmp.data());
    for (int i = 0; i < n; i++) tmp[i] = cx_sub(b[i], tmp[i]);
    apply_precond(tmp.data());
    relres = cvec_nrm2(n, tmp.data()) / normMb;
  } else {
    relres = beta / normMb;
  }

  auto result = Napi::Object::New(env);
  // Split x into re/im
  auto xReArr = Napi::Float64Array::New(env, n);
  auto xImArr = Napi::Float64Array::New(env, n);
  for (int i = 0; i < n; i++) { xReArr[i] = x[i].real; xImArr[i] = x[i].imag; }
  result.Set("xRe", xReArr);
  result.Set("xIm", xImArr);
  result.Set("flag", Napi::Number::New(env, flag));
  result.Set("relres", Napi::Number::New(env, relres));
  auto iterArr = Napi::Int32Array::New(env, 2);
  iterArr[0] = outerIter; iterArr[1] = innerIter;
  result.Set("iter", iterArr);
  result.Set("resvec", vecToF64(env, resvec));
  return result;
}
