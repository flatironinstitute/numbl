% size and ndims on 3D tensors

A = reshape(1:24, 2, 3, 4);

% ── size with dimension argument ─────────────────────────────────
assert(size(A, 1) == 2)
assert(size(A, 2) == 3)
assert(size(A, 3) == 4)

% ── size returns full shape vector ───────────────────────────────
s = size(A);
assert(length(s) == 3)
assert(s(1) == 2)
assert(s(2) == 3)
assert(s(3) == 4)

% ── multiple output: [m,n,p] = size(A) ──────────────────────────
[m, n, p] = size(A);
assert(m == 2)
assert(n == 3)
assert(p == 4)

% ── ndims ────────────────────────────────────────────────────────
assert(ndims(A) == 3)
assert(ndims([1 2 3]) == 2)  % row vector is 2D
assert(ndims(5) == 2)         % scalar is 2D

% ── numel on 3D ──────────────────────────────────────────────────
assert(numel(A) == 24)

disp('SUCCESS')
