% 3D tensor manipulation: fliplr, flipud, repmat

a = reshape(1:12, 2, 3, 2);

% ── fliplr on 3D: flip columns (dim 2) on each page ──────────────
b = fliplr(a);
assert(size(b, 1) == 2)
assert(size(b, 2) == 3)
assert(size(b, 3) == 2)
% Page 1: columns reversed
assert(b(1, 1, 1) == a(1, 3, 1))
assert(b(1, 3, 1) == a(1, 1, 1))
assert(b(2, 2, 1) == a(2, 2, 1))
% Page 2: columns reversed
assert(b(1, 1, 2) == a(1, 3, 2))

% ── flipud on 3D: flip rows (dim 1) on each page ─────────────────
c = flipud(a);
assert(size(c, 1) == 2)
assert(size(c, 2) == 3)
assert(size(c, 3) == 2)
% Page 1: rows reversed
assert(c(1, 1, 1) == a(2, 1, 1))
assert(c(2, 1, 1) == a(1, 1, 1))
% Page 2: rows reversed
assert(c(1, 1, 2) == a(2, 1, 2))

% ── repmat on 3D input ────────────────────────────────────────────
d = reshape(1:8, 2, 2, 2);
e = repmat(d, 2, 3);
assert(size(e, 1) == 4)
assert(size(e, 2) == 6)
assert(size(e, 3) == 2)
% Check tiling: first tile
assert(e(1, 1, 1) == d(1, 1, 1))
assert(e(2, 2, 1) == d(2, 2, 1))
% Check tiling: second row tile
assert(e(3, 1, 1) == d(1, 1, 1))
assert(e(4, 2, 1) == d(2, 2, 1))
% Check tiling: second col tile
assert(e(1, 3, 1) == d(1, 1, 1))
% Check page 2
assert(e(1, 1, 2) == d(1, 1, 2))
assert(e(3, 3, 2) == d(1, 1, 2))

disp('SUCCESS')
