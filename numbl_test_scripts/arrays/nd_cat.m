% 3D tensor concatenation: cat along dim 3

% ── cat(3, A, B) — stack two 2D matrices along 3rd dimension ────────
A = [1 2; 3 4];
B = [5 6; 7 8];
C = cat(3, A, B);
assert(size(C, 1) == 2)
assert(size(C, 2) == 2)
assert(size(C, 3) == 2)
assert(C(1, 1, 1) == 1)
assert(C(2, 2, 1) == 4)
assert(C(1, 1, 2) == 5)
assert(C(2, 2, 2) == 8)

% ── cat(3, A, B, C_mat) — three pages ───────────────────────────────
D = [9 10; 11 12];
E = cat(3, A, B, D);
assert(size(E, 3) == 3)
assert(E(1, 1, 3) == 9)
assert(E(2, 2, 3) == 12)

% ── cat(1, ...) and cat(2, ...) still work (regression) ─────────────
F = cat(1, [1 2], [3 4]);
assert(size(F, 1) == 2)
assert(F(2, 1) == 3)

G = cat(2, [1; 2], [3; 4]);
assert(size(G, 2) == 2)
assert(G(1, 2) == 3)

disp('SUCCESS')
