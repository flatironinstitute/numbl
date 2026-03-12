% 3D tensor element-wise arithmetic and broadcasting

a = reshape(1:24, 2, 3, 4);

% ── Same-shape element-wise ops ─────────────────────────────────────
b = a + a;
assert(b(1, 1, 1) == 2)
assert(b(2, 3, 4) == 48)

c = a .* a;
assert(c(1, 1, 1) == 1)
assert(c(2, 1, 1) == 4)

% ── Scalar broadcast ────────────────────────────────────────────────
d = a + 10;
assert(d(1, 1, 1) == 11)
assert(d(2, 3, 4) == 34)

e = 2 .* a;
assert(e(1, 1, 1) == 2)
assert(e(2, 3, 4) == 48)

% ── Broadcasting: [2,3,4] + [1,3,4] ────────────────────────────────
% Broadcast along dim 1
f = ones(1, 3, 4);
g = a + f;
assert(g(1, 1, 1) == 2)
assert(g(2, 1, 1) == 3)

% ── Broadcasting: [2,3,4] + [2,1,4] ────────────────────────────────
% Broadcast along dim 2
h = ones(2, 1, 4);
k = a + h;
assert(k(1, 1, 1) == 2)
assert(k(1, 2, 1) == 4)

% ── Broadcasting: [2,3,4] + [2,3,1] ────────────────────────────────
% Broadcast along dim 3
m = ones(2, 3, 1);
n = a + m;
assert(n(1, 1, 1) == 2)
assert(n(1, 1, 2) == 8)

disp('SUCCESS')
