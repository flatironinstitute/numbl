% Test besselh: Hankel function (Bessel function of the third kind)
tol = 1e-10;

% ---- Default K=1: H^(1)_nu(z) = J_nu(z) + i*Y_nu(z) ----
H = besselh(0, 1);
expected = besselj(0, 1) + 1i*bessely(0, 1);
assert(abs(H - expected) < tol);

H = besselh(1, 1);
expected = besselj(1, 1) + 1i*bessely(1, 1);
assert(abs(H - expected) < tol);

H = besselh(2, 3);
expected = besselj(2, 3) + 1i*bessely(2, 3);
assert(abs(H - expected) < tol);

% ---- K=1 explicit ----
H = besselh(0, 1, 2);
expected = besselj(0, 2) + 1i*bessely(0, 2);
assert(abs(H - expected) < tol);

% ---- K=2: H^(2)_nu(z) = J_nu(z) - i*Y_nu(z) ----
H = besselh(0, 2, 2);
expected = besselj(0, 2) - 1i*bessely(0, 2);
assert(abs(H - expected) < tol);

H = besselh(1, 2, 1);
expected = besselj(1, 1) - 1i*bessely(1, 1);
assert(abs(H - expected) < tol);

% ---- Non-integer order ----
H = besselh(0.5, 1);
expected = besselj(0.5, 1) + 1i*bessely(0.5, 1);
assert(abs(H - expected) < tol);

% ---- Vector Z input ----
z = [1, 2, 3];
H = besselh(0, z);
for kk = 1:3
  exp_k = besselj(0, z(kk)) + 1i*bessely(0, z(kk));
  assert(abs(H(kk) - exp_k) < tol);
end

% ---- Vector NU input ----
nu = [0, 1, 2];
H = besselh(nu, 2);
for kk = 1:3
  exp_k = besselj(nu(kk), 2) + 1i*bessely(nu(kk), 2);
  assert(abs(H(kk) - exp_k) < tol);
end

% ---- Vector NU with K=2 ----
H = besselh(nu, 2, 2);
for kk = 1:3
  exp_k = besselj(nu(kk), 2) - 1i*bessely(nu(kk), 2);
  assert(abs(H(kk) - exp_k) < tol);
end

% ---- Scaling: K=1, scaled by e^(-i*z) ----
H = besselh(0, 1, 1, 1);
expected = (besselj(0, 1) + 1i*bessely(0, 1)) * exp(-1i*1);
assert(abs(H - expected) < tol);

H = besselh(1, 1, 2.5, 1);
expected = (besselj(1, 2.5) + 1i*bessely(1, 2.5)) * exp(-1i*2.5);
assert(abs(H - expected) < tol);

% ---- Scaling: K=2, scaled by e^(+i*z) ----
H = besselh(0, 2, 1, 1);
expected = (besselj(0, 1) - 1i*bessely(0, 1)) * exp(1i*1);
assert(abs(H - expected) < tol);

H = besselh(1, 2, 2.5, 1);
expected = (besselj(1, 2.5) - 1i*bessely(1, 2.5)) * exp(1i*2.5);
assert(abs(H - expected) < tol);

% ---- Unscaled (scale=0) explicit ----
H = besselh(0, 1, 1, 0);
expected = besselj(0, 1) + 1i*bessely(0, 1);
assert(abs(H - expected) < tol);

disp('SUCCESS');
