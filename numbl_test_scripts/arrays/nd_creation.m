% 3D tensor creation: zeros, ones, reshape, size, ndims

% ── zeros with 3 dimensions ─────────────────────────────────────────
a = zeros(2, 3, 4);
assert(ndims(a) == 3)
assert(size(a, 1) == 2)
assert(size(a, 2) == 3)
assert(size(a, 3) == 4)
assert(numel(a) == 24)

% size returns full shape vector
s = size(a);
assert(length(s) == 3)
assert(s(1) == 2)
assert(s(2) == 3)
assert(s(3) == 4)

% ── ones with 3 dimensions ──────────────────────────────────────────
b = ones(2, 3, 4);
assert(ndims(b) == 3)
assert(numel(b) == 24)

% all elements should be 1
assert(sum(b(:)) == 24)

% ── reshape to 3D ───────────────────────────────────────────────────
v = 1:24;
c = reshape(v, 2, 3, 4);
assert(ndims(c) == 3)
assert(size(c, 1) == 2)
assert(size(c, 2) == 3)
assert(size(c, 3) == 4)
assert(numel(c) == 24)

% ── reshape back to 2D ──────────────────────────────────────────────
d = reshape(c, 6, 4);
assert(ndims(d) == 2)
assert(size(d, 1) == 6)
assert(size(d, 2) == 4)

% ── reshape to 1D column ────────────────────────────────────────────
e = reshape(c, 24, 1);
assert(size(e, 1) == 24)
assert(size(e, 2) == 1)

% ── zeros with vector arg ───────────────────────────────────────────
f = zeros([2, 3, 4]);
assert(ndims(f) == 3)
assert(size(f, 1) == 2)
assert(size(f, 2) == 3)
assert(size(f, 3) == 4)

% ── length of 3D array = max dimension ──────────────────────────────
assert(length(a) == 4)

disp('SUCCESS')
