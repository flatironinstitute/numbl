% Broadcasting arithmetic: scalar with matrix/3D, vector with matrix

% ── scalar + matrix ──────────────────────────────────────────────
A = [1 2; 3 4];
B = A + 10;
assert(B(1,1) == 11)
assert(B(2,2) == 14)

% ── scalar * 3D tensor ──────────────────────────────────────────
C = reshape(1:8, 2, 2, 2);
D = 2 * C;
assert(D(1,1,1) == 2)
assert(D(2,2,2) == 16)

% ── scalar ./ 3D tensor ─────────────────────────────────────────
E = 12 ./ C;
assert(E(1,1,1) == 12)
assert(abs(E(2,1,1) - 6) < 1e-10)

% ── matrix - scalar ─────────────────────────────────────────────
F = A - 1;
assert(F(1,1) == 0)
assert(F(2,2) == 3)

% ── row vector + column vector → matrix (broadcast) ─────────────
r = [1 2 3];
c = [10; 20];
G = r + c;
% Should be [2,3]: [11 12 13; 21 22 23]
assert(size(G, 1) == 2)
assert(size(G, 2) == 3)
assert(G(1,1) == 11)
assert(G(2,3) == 23)

disp('SUCCESS')
