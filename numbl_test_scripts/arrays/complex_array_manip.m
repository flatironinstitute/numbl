% Complex array manipulation: flip, repmat, permute, transpose, diag

% ── fliplr on complex ──────────────────────────────────────────────
g = fliplr([1+1i, 2+2i, 3+3i]);
assert(abs(real(g(1)) - 3) < 1e-10)
assert(abs(imag(g(1)) - 3) < 1e-10)
assert(abs(real(g(3)) - 1) < 1e-10)
assert(abs(imag(g(3)) - 1) < 1e-10)

% ── flipud on complex ──────────────────────────────────────────────
h = flipud([1+1i; 2+2i; 3+3i]);
assert(abs(real(h(1)) - 3) < 1e-10)
assert(abs(imag(h(1)) - 3) < 1e-10)
assert(abs(real(h(3)) - 1) < 1e-10)
assert(abs(imag(h(3)) - 1) < 1e-10)

% ── fliplr on complex matrix ───────────────────────────────────────
M = [1+1i 2+2i; 3+3i 4+4i];
Mf = fliplr(M);
assert(abs(real(Mf(1,1)) - 2) < 1e-10)
assert(abs(imag(Mf(1,1)) - 2) < 1e-10)
assert(abs(real(Mf(2,2)) - 3) < 1e-10)
assert(abs(imag(Mf(2,2)) - 3) < 1e-10)

% ── repmat on complex ──────────────────────────────────────────────
r = repmat([1+2i 3+4i], 1, 2);
assert(length(r) == 4)
assert(abs(real(r(3)) - 1) < 1e-10)
assert(abs(imag(r(3)) - 2) < 1e-10)
assert(abs(real(r(4)) - 3) < 1e-10)
assert(abs(imag(r(4)) - 4) < 1e-10)

% ── repmat complex matrix ──────────────────────────────────────────
R = repmat([1+1i; 2+2i], 2, 1);
assert(size(R, 1) == 4)
assert(abs(real(R(3,1)) - 1) < 1e-10)
assert(abs(imag(R(3,1)) - 1) < 1e-10)

% ── permute on complex ─────────────────────────────────────────────
P = reshape([1+1i 2+2i 3+3i 4+4i 5+5i 6+6i], 2, 3);
S = permute(P, [2 1]);
assert(size(S, 1) == 3)
assert(size(S, 2) == 2)
assert(abs(real(S(2,1)) - 3) < 1e-10)
assert(abs(imag(S(2,1)) - 3) < 1e-10)

% ── transpose (.' operator) on complex ─────────────────────────────
T = [1+2i 3+4i; 5+6i 7+8i];
Tt = T.';
assert(abs(real(Tt(1,2)) - 5) < 1e-10)
assert(abs(imag(Tt(1,2)) - 6) < 1e-10)
assert(abs(real(Tt(2,1)) - 3) < 1e-10)
assert(abs(imag(Tt(2,1)) - 4) < 1e-10)

% ── ctranspose (' operator) on complex ─────────────────────────────
Tc = T';
assert(abs(real(Tc(1,2)) - 5) < 1e-10)
assert(abs(imag(Tc(1,2)) - (-6)) < 1e-10)
assert(abs(real(Tc(2,1)) - 3) < 1e-10)
assert(abs(imag(Tc(2,1)) - (-4)) < 1e-10)

% ── diag: complex vector → matrix ──────────────────────────────────
D = diag([1+2i, 3+4i, 5+6i]);
assert(size(D, 1) == 3)
assert(size(D, 2) == 3)
assert(abs(real(D(1,1)) - 1) < 1e-10)
assert(abs(imag(D(1,1)) - 2) < 1e-10)
assert(abs(real(D(2,2)) - 3) < 1e-10)
assert(abs(imag(D(2,2)) - 4) < 1e-10)
assert(D(1,2) == 0)

% ── diag: complex matrix → extract diagonal ────────────────────────
M2 = [1+2i 3+4i; 5+6i 7+8i];
dv = diag(M2);
assert(length(dv) == 2)
assert(abs(real(dv(1)) - 1) < 1e-10)
assert(abs(imag(dv(1)) - 2) < 1e-10)
assert(abs(real(dv(2)) - 7) < 1e-10)
assert(abs(imag(dv(2)) - 8) < 1e-10)

disp('SUCCESS')
