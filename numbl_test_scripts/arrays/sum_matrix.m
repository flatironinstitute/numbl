% sum/prod/mean on matrices should reduce along dim 1

% ── sum on matrix: column sums → row vector ──────────────────────
A = [1 3; 2 4];
B = sum(A);
% Column 1: 1+2=3.  Column 2: 3+4=7
assert(length(B) == 2)
assert(B(1) == 3)
assert(B(2) == 7)

% ── sum on row vector stays scalar ───────────────────────────────
assert(sum([1 2 3]) == 6)

% ── sum on column vector → scalar ────────────────────────────────
assert(sum([1; 2; 3]) == 6)

% ── prod on matrix: column products → row vector ─────────────────
C = [2 3; 4 5];
D = prod(C);
% Column 1: 2*4=8.  Column 2: 3*5=15
assert(length(D) == 2)
assert(D(1) == 8)
assert(D(2) == 15)

% ── mean on matrix: column means → row vector ────────────────────
E = [1 5; 3 7];
F = mean(E);
% Column 1: (1+3)/2=2.  Column 2: (5+7)/2=6
assert(length(F) == 2)
assert(F(1) == 2)
assert(F(2) == 6)

disp('SUCCESS')
