% Comprehensive complex tensor tests: assignment, reshape, indexing, reductions

% ═══════════════════════════════════════════════════════════════════
% Assignment
% ═══════════════════════════════════════════════════════════════════

% ── 2D scalar assignment into complex matrix ─────────────────────
M = [1+1i 2+2i; 3+3i 4+4i];
M(1,2) = 10+20i;
assert(abs(real(M(1,2)) - 10) < 1e-10)
assert(abs(imag(M(1,2)) - 20) < 1e-10)
assert(real(M(2,2)) == 4)

% ── Assign real into complex matrix (zeroes imag) ────────────────
M(2,1) = 99;
assert(real(M(2,1)) == 99)
assert(imag(M(2,1)) == 0)

% ── Assign complex into real matrix (promotes to complex) ────────
R = [1 2; 3 4];
R(1,1) = 5+6i;
assert(abs(real(R(1,1)) - 5) < 1e-10)
assert(abs(imag(R(1,1)) - 6) < 1e-10)
assert(R(2,2) == 4)

% ── Column assignment with complex vector ────────────────────────
A = zeros(2, 2);
A(:,1) = [1+2i; 3+4i];
assert(abs(real(A(1,1)) - 1) < 1e-10)
assert(abs(imag(A(1,1)) - 2) < 1e-10)
assert(abs(real(A(2,1)) - 3) < 1e-10)
assert(abs(imag(A(2,1)) - 4) < 1e-10)

% ── Colon assign zeros complex vector ────────────────────────────
v = [1+1i, 2+2i, 3+3i];
v(:) = 0;
assert(real(v(1)) == 0)
assert(imag(v(1)) == 0)

% ── 1D index assign ──────────────────────────────────────────────
w = zeros(1, 4);
w(3) = 7+8i;
assert(abs(real(w(3)) - 7) < 1e-10)
assert(abs(imag(w(3)) - 8) < 1e-10)
assert(w(1) == 0)

% ═══════════════════════════════════════════════════════════════════
% Reshape preserves complex
% ═══════════════════════════════════════════════════════════════════

F = reshape([1+1i 2+2i 3+3i 4+4i 5+5i 6+6i], 2, 3);
assert(size(F, 1) == 2)
assert(size(F, 2) == 3)
assert(abs(real(F(1,1)) - 1) < 1e-10)
assert(abs(imag(F(1,1)) - 1) < 1e-10)
assert(abs(real(F(2,3)) - 6) < 1e-10)
assert(abs(imag(F(2,3)) - 6) < 1e-10)

% ═══════════════════════════════════════════════════════════════════
% Logical indexing on complex
% ═══════════════════════════════════════════════════════════════════

G = [1+0i 2+3i 4+0i 0+5i];
H = G(abs(G) > 3);
assert(length(H) == 3)
assert(abs(real(H(1)) - 2) < 1e-10)
assert(abs(imag(H(1)) - 3) < 1e-10)
assert(abs(real(H(2)) - 4) < 1e-10)
assert(abs(imag(H(3)) - 5) < 1e-10)

% ═══════════════════════════════════════════════════════════════════
% Mixed real/complex arithmetic
% ═══════════════════════════════════════════════════════════════════

B2 = [1 2; 3 4] + 1i;
assert(abs(real(B2(2,2)) - 4) < 1e-10)
assert(abs(imag(B2(2,2)) - 1) < 1e-10)

C2 = B2 + 10;
assert(abs(real(C2(1,1)) - 11) < 1e-10)
assert(abs(imag(C2(1,1)) - 1) < 1e-10)

D2 = B2 * 2;
assert(abs(real(D2(1,1)) - 2) < 1e-10)
assert(abs(imag(D2(1,1)) - 2) < 1e-10)

% ═══════════════════════════════════════════════════════════════════
% Complex reductions: prod, sort
% ═══════════════════════════════════════════════════════════════════

% ── prod ─────────────────────────────────────────────────────────
p = prod([1+2i, 3+4i]);
assert(abs(real(p) - (-5)) < 1e-10)
assert(abs(imag(p) - 10) < 1e-10)

% ── prod on matrix ───────────────────────────────────────────────
PM = [1+1i 2+0i; 0+1i 3+1i];
pm = prod(PM);
assert(abs(real(pm(1)) - (-1)) < 1e-10)
assert(abs(imag(pm(1)) - 1) < 1e-10)
assert(abs(real(pm(2)) - 6) < 1e-10)
assert(abs(imag(pm(2)) - 2) < 1e-10)

% ── sort by magnitude ────────────────────────────────────────────
sv = sort([3+4i, 1+0i, 0+2i]);
assert(abs(real(sv(1)) - 1) < 1e-10)
assert(abs(real(sv(2)) - 0) < 1e-10)
assert(abs(imag(sv(2)) - 2) < 1e-10)
assert(abs(real(sv(3)) - 3) < 1e-10)
assert(abs(imag(sv(3)) - 4) < 1e-10)

% ═══════════════════════════════════════════════════════════════════
% Conjugate transpose and matrix multiply
% ═══════════════════════════════════════════════════════════════════

% ── conj on matrix ───────────────────────────────────────────────
I2 = conj([1+2i 3+4i; 5+6i 7+8i]);
assert(imag(I2(1,1)) == -2)
assert(imag(I2(2,2)) == -8)

% ── Complex matmul ───────────────────────────────────────────────
AA = [1+1i 2; 0 1-1i];
xx = [1+0i; 0+1i];
bb = AA * xx;
assert(abs(real(bb(1)) - 1) < 1e-10)
assert(abs(imag(bb(1)) - 3) < 1e-10)
assert(abs(real(bb(2)) - 1) < 1e-10)
assert(abs(imag(bb(2)) - 1) < 1e-10)

% ── Hermitian: M'*M diagonal is real ─────────────────────────────
MM = [1+2i 3+4i; 5+6i 7+8i];
HH = MM' * MM;
assert(abs(imag(HH(1,1))) < 1e-10)
assert(abs(imag(HH(2,2))) < 1e-10)
assert(abs(real(HH(1,1)) - 66) < 1e-10)

disp('SUCCESS')
