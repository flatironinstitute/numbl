% Logical indexed assignment

% ── Scalar assignment with logical mask ────────────────────────────
v = [1 2 3 4 5];
v(v > 3) = 0;
assert(v(1) == 1)
assert(v(2) == 2)
assert(v(3) == 3)
assert(v(4) == 0)
assert(v(5) == 0)

% ── Replace negatives with zero ────────────────────────────────────
w = [3 -1 4 -2 5];
w(w < 0) = 0;
assert(w(1) == 3)
assert(w(2) == 0)
assert(w(3) == 4)
assert(w(4) == 0)
assert(w(5) == 5)

% ── Logical mask on matrix ─────────────────────────────────────────
M = [1 2; 3 4];
M(M > 2) = 99;
assert(M(1,1) == 1)
assert(M(2,1) == 99)
assert(M(1,2) == 2)
assert(M(2,2) == 99)

% ── Assign vector via logical mask ─────────────────────────────────
x = [10 20 30 40 50];
mask = x >= 30;
x(mask) = [0 0 0];
assert(x(1) == 10)
assert(x(2) == 20)
assert(x(3) == 0)
assert(x(4) == 0)
assert(x(5) == 0)

% ── Scalar base with 2D logical column index ─────────────────────────
% A(:, true) = 99 should assign to column 1 of a scalar
A = 5;
A(:, true) = 99;
assert(A == 99)

% ── Matrix with 2D logical column index ──────────────────────────────
B = [1 2 3; 4 5 6];
B(:, [true false true]) = 0;
assert(B(1,1) == 0)
assert(B(2,1) == 0)
assert(B(1,2) == 2)
assert(B(2,2) == 5)
assert(B(1,3) == 0)
assert(B(2,3) == 0)

% ── 1x1 tensor with 2D logical column index ─────────────────────────
C = [7];
C(:, true) = 42;
assert(C == 42)

% ── Scalar base with logical true from comparison ────────────────────
D = 10;
flag = D > 5;
D(:, flag) = real(D(:, flag));
assert(D == 10)

disp('SUCCESS')
