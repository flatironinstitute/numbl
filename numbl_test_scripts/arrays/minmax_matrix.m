% min/max on matrices should reduce along dim 1

% ── min on matrix: column mins → row vector ──────────────────────
A = [3 1; 2 4; 5 0];
B = min(A);
% Column 1: min([3;2;5]) = 2.  Column 2: min([1;4;0]) = 0
assert(length(B) == 2)
assert(B(1) == 2)
assert(B(2) == 0)

% ── max on matrix: column maxes → row vector ─────────────────────
C = max(A);
% Column 1: max([3;2;5]) = 5.  Column 2: max([1;4;0]) = 4
assert(length(C) == 2)
assert(C(1) == 5)
assert(C(2) == 4)

% ── min/max on row vector stays scalar ───────────────────────────
assert(min([3 1 2]) == 1)
assert(max([3 1 2]) == 3)

% ── min/max on column vector → scalar ────────────────────────────
assert(min([3; 1; 2]) == 1)
assert(max([3; 1; 2]) == 3)

% ── min on 3D tensor: reduce along dim 1 ────────────────────────
% D is [2,3,2] with values 1..12
D = reshape(1:12, 2, 3, 2);
E = min(D);
% Result should be [1,3,2]: min of each column along dim 1
% Page 1: cols [1;2]->[1], [3;4]->[3], [5;6]->[5]
% Page 2: cols [7;8]->[7], [9;10]->[9], [11;12]->[11]
assert(E(1,1,1) == 1)
assert(E(1,2,1) == 3)
assert(E(1,3,1) == 5)
assert(E(1,1,2) == 7)
assert(E(1,3,2) == 11)

% ── max on 3D tensor ────────────────────────────────────────────
F = max(D);
assert(F(1,1,1) == 2)
assert(F(1,2,1) == 4)
assert(F(1,3,1) == 6)
assert(F(1,1,2) == 8)
assert(F(1,3,2) == 12)

disp('SUCCESS')
