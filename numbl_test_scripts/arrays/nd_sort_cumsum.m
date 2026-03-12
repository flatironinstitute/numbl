% sort and cumsum should operate along dim 1 by default

% ── sort on 2D matrix: sorts each column independently ────────────
A = [3 1; 2 4];
B = sort(A);
% Column 1: sort [3;2] → [2;3]. Column 2: sort [1;4] → [1;4]
assert(B(1,1) == 2)
assert(B(2,1) == 3)
assert(B(1,2) == 1)
assert(B(2,2) == 4)

% ── cumsum on 2D matrix: cumsum along each column ─────────────────
C = [1 3; 2 4];
D = cumsum(C);
% Column 1: cumsum [1;2] → [1;3]. Column 2: cumsum [3;4] → [3;7]
assert(D(1,1) == 1)
assert(D(2,1) == 3)
assert(D(1,2) == 3)
assert(D(2,2) == 7)

% ── sort on 3D ────────────────────────────────────────────────────
E = reshape([6 5 4 3 2 1 12 11 10 9 8 7], 2, 3, 2);
F = sort(E);
% Each column of each page sorted independently
% Page 1 col 1: sort [6;5] → [5;6]
assert(F(1,1,1) == 5)
assert(F(2,1,1) == 6)

% ── cumsum on 3D ──────────────────────────────────────────────────
G = reshape(1:8, 2, 2, 2);
H = cumsum(G);
% Column 1 page 1: cumsum [1;2] → [1;3]
assert(H(1,1,1) == 1)
assert(H(2,1,1) == 3)
% Column 2 page 1: cumsum [3;4] → [3;7]
assert(H(1,2,1) == 3)
assert(H(2,2,1) == 7)
% Column 1 page 2: cumsum [5;6] → [5;11]
assert(H(1,1,2) == 5)
assert(H(2,1,2) == 11)

disp('SUCCESS')
