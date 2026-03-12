% any/all should operate along dim 1 for matrices (not reduce to scalar)

% ── any on matrix: returns row vector ────────────────────────────
A = [0 1; 0 0; 1 0];
B = any(A);
% Column 1: any([0;0;1]) = 1.  Column 2: any([1;0;0]) = 1
assert(length(B) == 2)
assert(B(1) == 1)
assert(B(2) == 1)

% ── any with a zero column ──────────────────────────────────────
C = [0 1; 0 0];
D = any(C);
% Column 1: any([0;0]) = 0.  Column 2: any([1;0]) = 1
assert(D(1) == 0)
assert(D(2) == 1)

% ── all on matrix: returns row vector ────────────────────────────
E = [1 1; 1 0; 1 1];
F = all(E);
% Column 1: all([1;1;1]) = 1.  Column 2: all([1;0;1]) = 0
assert(length(F) == 2)
assert(F(1) == 1)
assert(F(2) == 0)

% ── any on row vector stays scalar ───────────────────────────────
assert(any([0 0 1]) == 1)
assert(any([0 0 0]) == 0)

% ── all on row vector stays scalar ───────────────────────────────
assert(all([1 1 1]) == 1)
assert(all([1 1 0]) == 0)

% ── any/all on column vector should return scalar ────────────────
v = [1; 0; 1];
assert(any(v) == 1)
assert(all(v) == 0)

% ── any/all on column vector in boolean context ──────────────────
if any(v)
    x = 1;
else
    x = 0;
end
assert(x == 1)

disp('SUCCESS')
