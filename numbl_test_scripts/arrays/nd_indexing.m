% 3D tensor indexing: scalar, colon, and range indexing

% Create a 2x3x4 tensor with known values
% In MATLAB, reshape fills column-major: varying dim1 fastest, then dim2, then dim3
v = 1:24;
a = reshape(v, 2, 3, 4);

% ── Scalar indexing: a(r, c, p) ─────────────────────────────────────
% Column-major order: a(1,1,1)=1, a(2,1,1)=2, a(1,2,1)=3, a(2,2,1)=4, ...
assert(a(1, 1, 1) == 1)
assert(a(2, 1, 1) == 2)
assert(a(1, 2, 1) == 3)
assert(a(2, 2, 1) == 4)
assert(a(1, 3, 1) == 5)
assert(a(2, 3, 1) == 6)

% Second page (3rd dim = 2)
assert(a(1, 1, 2) == 7)
assert(a(2, 1, 2) == 8)

% Last element
assert(a(2, 3, 4) == 24)

% ── Colon in one dimension: a(:, 1, 1) ──────────────────────────────
b = a(:, 1, 1);
assert(numel(b) == 2)
assert(b(1) == 1)
assert(b(2) == 2)

% ── Colon in another dimension: a(1, :, 1) ──────────────────────────
c = a(1, :, 1);
assert(numel(c) == 3)
assert(c(1) == 1)
assert(c(2) == 3)
assert(c(3) == 5)

% ── Colon in third dimension: a(1, 1, :) ────────────────────────────
d = a(1, 1, :);
assert(numel(d) == 4)
assert(d(1) == 1)
assert(d(2) == 7)
assert(d(3) == 13)
assert(d(4) == 19)

% ── Two colons: a(:, :, 1) — first "page" ───────────────────────────
e = a(:, :, 1);
assert(size(e, 1) == 2)
assert(size(e, 2) == 3)
assert(e(1, 1) == 1)
assert(e(2, 1) == 2)
assert(e(1, 3) == 5)
assert(e(2, 3) == 6)

% ── All colons: a(:, :, :) — copy ───────────────────────────────────
h = a(:, :, :);
assert(numel(h) == 24)
assert(h(1, 1, 1) == 1)
assert(h(2, 3, 4) == 24)

disp('SUCCESS')
