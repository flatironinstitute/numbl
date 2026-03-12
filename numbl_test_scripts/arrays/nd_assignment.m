% 3D tensor indexed assignment

% ── Scalar assignment: a(r, c, p) = val ─────────────────────────────
a = zeros(2, 3, 4);
a(1, 1, 1) = 100;
assert(a(1, 1, 1) == 100)
assert(a(2, 1, 1) == 0)

a(2, 3, 4) = 999;
assert(a(2, 3, 4) == 999)
assert(a(1, 3, 4) == 0)

% ── Colon assignment: a(:, 1, 1) = [10; 20] ─────────────────────────
b = zeros(2, 3, 4);
b(:, 1, 1) = [10; 20];
assert(b(1, 1, 1) == 10)
assert(b(2, 1, 1) == 20)
assert(b(1, 2, 1) == 0)

% ── Page assignment: a(:, :, 2) = matrix ─────────────────────────────
c = zeros(2, 3, 4);
c(:, :, 2) = [1 2 3; 4 5 6];
assert(c(1, 1, 2) == 1)
assert(c(2, 1, 2) == 4)
assert(c(1, 3, 2) == 3)
assert(c(2, 3, 2) == 6)
assert(c(1, 1, 1) == 0)  % other pages unchanged

disp('SUCCESS')
