% 3D tensor: advanced indexing patterns

a = reshape(1:24, 2, 3, 4);

% ── Range indexing in multiple dims simultaneously ────────────────
b = a(1:2, 2:3, 2:3);
assert(size(b, 1) == 2)
assert(size(b, 2) == 2)
assert(size(b, 3) == 2)
assert(b(1, 1, 1) == a(1, 2, 2))
assert(b(2, 2, 2) == a(2, 3, 3))

% ── Extracting a single row across all cols and pages ─────────────
c = a(1, :, :);
assert(size(c, 1) == 1)
assert(size(c, 2) == 3)
assert(size(c, 3) == 4)
assert(c(1, 1, 1) == 1)
assert(c(1, 2, 3) == a(1, 2, 3))

% ── Extracting a single column across all rows and pages ──────────
d = a(:, 2, :);
assert(size(d, 1) == 2)
assert(size(d, 2) == 1)
assert(size(d, 3) == 4)
assert(d(1, 1, 1) == a(1, 2, 1))
assert(d(2, 1, 4) == a(2, 2, 4))

% ── Assigning to a slice with ranges ──────────────────────────────
e = zeros(2, 3, 2);
e(1:2, 1:2, 1) = [10 20; 30 40];
assert(e(1, 1, 1) == 10)
assert(e(2, 1, 1) == 30)
assert(e(1, 2, 1) == 20)
assert(e(2, 2, 1) == 40)
assert(e(1, 3, 1) == 0)
assert(e(1, 1, 2) == 0)

% ── Linear indexing on 3D ─────────────────────────────────────────
% MATLAB linear indexing is column-major
assert(a(1) == 1)
assert(a(2) == 2)
assert(a(3) == 3)   % a(2,1,1) + 1 in linear order
assert(a(7) == 7)   % start of page 2
assert(a(24) == 24)

% ── Vector index into 3D ─────────────────────────────────────────
f = a([1 3 5]);
assert(f(1) == 1)
assert(f(2) == 3)
assert(f(3) == 5)

disp('SUCCESS')
