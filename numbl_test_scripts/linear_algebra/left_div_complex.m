% Test complex left division (mldivide) and linsolve with complex matrices
tol = 1e-10;

% ── Square system: complex A, real B ─────────────────────────────────────────
% A = [1+i, 0; 0, 2-i], b = [1+i; 2-i]  →  x = [1; 1]
A = [1+1i, 0; 0, 2-1i];
b = [1+1i; 2-1i];
x = A \ b;
assert(abs(real(x(1)) - 1) < tol)
assert(abs(imag(x(1)))     < tol)
assert(abs(real(x(2)) - 1) < tol)
assert(abs(imag(x(2)))     < tol)

% Verify A * x == b
res = A * x - b;
assert(abs(real(res(1))) < tol)
assert(abs(imag(res(1))) < tol)
assert(abs(real(res(2))) < tol)
assert(abs(imag(res(2))) < tol)

% ── Square system: real A, complex B ─────────────────────────────────────────
% A = [2 1; 5 3],  b = [8+2i; 22+5i]  →  x = [2+1i; 4]
Ar = [2, 1; 5, 3];
bc = [8+2i; 22+5i];
xc = Ar \ bc;
assert(abs(real(xc(1)) - 2) < tol)
assert(abs(imag(xc(1)) - 1) < tol)
assert(abs(real(xc(2)) - 4) < tol)
assert(abs(imag(xc(2)))     < tol)

% ── Square system: complex A, complex B ──────────────────────────────────────
% A = [2+1i, 1; 5, 3-1i], b = [...]
A2 = [2+1i, 1; 5, 3-1i];
b2 = [3+1i; 2-3i];
x2 = A2 \ b2;
% Verify A2 * x2 == b2
res2 = A2 * x2 - b2;
assert(abs(res2(1)) < tol)
assert(abs(res2(2)) < tol)

% \ and linsolve must agree
xl = linsolve(A2, b2);
assert(abs(xl(1) - x2(1)) < tol)
assert(abs(xl(2) - x2(2)) < tol)

% ── Multiple right-hand sides ─────────────────────────────────────────────────
B2 = [3+1i, 1; 2-3i, 2i];
X2 = A2 \ B2;
RES2 = A2 * X2 - B2;
assert(abs(RES2(1,1)) < tol)
assert(abs(RES2(2,1)) < tol)
assert(abs(RES2(1,2)) < tol)
assert(abs(RES2(2,2)) < tol)

% ── Overdetermined complex system (least-squares via QR) ─────────────────────
Ao = [1+1i; 2; 1-1i];    % 3×1 complex
bo = [3+1i; 4; 3-1i];    % 3×1 complex
xo = Ao \ bo;             % scalar least-squares solution
% Normal equations: Ao' * (Ao * xo - bo) ≈ 0
nr = Ao' * (Ao * xo - bo);
assert(abs(nr) < 1e-8)

disp('disable-for-now')
