% Reductions with explicit dimension argument on 3D tensors

A = reshape(1:12, 2, 3, 2);

% ── sum along dim 1: [2,3,2] → [1,3,2] ─────────────────────────
B = sum(A, 1);
assert(size(B, 1) == 1)
assert(size(B, 2) == 3)
assert(size(B, 3) == 2)
% Page 1 col 1: 1+2=3, col 2: 3+4=7, col 3: 5+6=11
assert(B(1,1,1) == 3)
assert(B(1,2,1) == 7)
assert(B(1,3,1) == 11)
% Page 2 col 1: 7+8=15
assert(B(1,1,2) == 15)

% ── sum along dim 2: [2,3,2] → [2,1,2] ─────────────────────────
C = sum(A, 2);
assert(size(C, 1) == 2)
assert(size(C, 2) == 1)
assert(size(C, 3) == 2)
% Row 1 page 1: 1+3+5=9, Row 2 page 1: 2+4+6=12
assert(C(1,1,1) == 9)
assert(C(2,1,1) == 12)
% Row 1 page 2: 7+9+11=27
assert(C(1,1,2) == 27)

% ── sum along dim 3: [2,3,2] → [2,3] ───────────────────────────
D = sum(A, 3);
assert(size(D, 1) == 2)
assert(size(D, 2) == 3)
% (1,1): 1+7=8, (2,1): 2+8=10, (1,2): 3+9=12
assert(D(1,1) == 8)
assert(D(2,1) == 10)
assert(D(1,2) == 12)
assert(D(2,3) == 18)

disp('SUCCESS')
