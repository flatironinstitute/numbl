% Test the \ (left division / mldivide) operator and linsolve equivalence
tol = 1e-6;

% ── Scalar left division ──────────────────────────────────────────────────────
% a \ b  =  b / a
r = 3 \ 12;
assert(abs(r - 4) < tol)

r2 = 0.5 \ 1;
assert(abs(r2 - 2) < tol)

% ── Element-wise left division ────────────────────────────────────────────────
% A .\ B  =  element-wise B / A
u = [2; 4; 5];
v = [6; 8; 15];
w = u .\ v;
assert(abs(w(1) - 3) < tol)
assert(abs(w(2) - 2) < tol)
assert(abs(w(3) - 3) < tol)

% ── Square system  A * x = b  (exact solution via LU) ────────────────────────
% A = [2 1; 5 3], b = [8; 22]  →  x = [2; 4]
A = [2, 1; 5, 3];
b = [8; 22];
x = A \ b;
assert(abs(x(1) - 2) < tol)
assert(abs(x(2) - 4) < tol)

% Verify A * x == b
res = A * x - b;
assert(abs(res(1)) < tol)
assert(abs(res(2)) < tol)

% \ must give the same result as linsolve(A, b)
xl = linsolve(A, b);
assert(abs(xl(1) - x(1)) < tol)
assert(abs(xl(2) - x(2)) < tol)

% ── Square system (3×3 diagonal) ─────────────────────────────────────────────
A3 = [1, 0, 0; 0, 2, 0; 0, 0, 3];
b3 = [1; 4; 9];
x3 = A3 \ b3;           % expect [1; 2; 3]
assert(abs(x3(1) - 1) < tol)
assert(abs(x3(2) - 2) < tol)
assert(abs(x3(3) - 3) < tol)

% ── Multiple right-hand sides  A * X = B ─────────────────────────────────────
% A = [2 1; 5 3],  B = [8 1; 22 4]
% Col 1 → [2; 4],  Col 2 → [-1; 3]
B = [8, 1; 22, 4];
X = A \ B;
assert(abs(X(1,1) - 2)  < tol)
assert(abs(X(2,1) - 4)  < tol)
assert(abs(X(1,2) - (-1)) < tol)
assert(abs(X(2,2) - 3)  < tol)

% Verify A * X == B
RES = A * X - B;
assert(abs(RES(1,1)) < tol)
assert(abs(RES(2,1)) < tol)
assert(abs(RES(1,2)) < tol)
assert(abs(RES(2,2)) < tol)

% ── Overdetermined system  (least-squares via QR) ────────────────────────────
% A is 3×2,  b is 3×1  →  x minimises ||A*x - b||
Ao = [1, 1; 1, 2; 1, 3];
bo = [1; 2; 2];
xo = Ao \ bo;         % expect ≈ [2/3; 1/2]
assert(abs(xo(1) - 2/3) < 1e-4)
assert(abs(xo(2) - 1/2) < 1e-4)

% Normal equations must be satisfied: A'*(A*x - b) ≈ 0
nr = Ao' * (Ao * xo - bo);
assert(abs(nr(1)) < 1e-4)
assert(abs(nr(2)) < 1e-4)

% ── Scalar A with vector B (scalar \ vec = vec / scalar) ─────────────────────
sv = 2 \ [4; 6; 10];
assert(abs(sv(1) - 2)  < tol)
assert(abs(sv(2) - 3)  < tol)
assert(abs(sv(3) - 5)  < tol)

disp('SUCCESS')
