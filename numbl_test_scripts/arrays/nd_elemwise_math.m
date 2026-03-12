% 3D tensor: element-wise math functions

a = reshape(1:8, 2, 2, 2);

% ── abs on 3D ─────────────────────────────────────────────────────
b = abs(-a);
assert(b(1,1,1) == 1)
assert(b(2,2,2) == 8)

% ── sqrt on 3D ────────────────────────────────────────────────────
c = sqrt(a);
assert(c(1,1,1) == 1)
assert(abs(c(2,2,1) - 2) < 1e-10)  % sqrt(4) = 2

% ── exp and log roundtrip ─────────────────────────────────────────
d = exp(log(a));
assert(abs(d(1,1,1) - 1) < 1e-10)
assert(abs(d(2,2,2) - 8) < 1e-10)

% ── power .^ on 3D ───────────────────────────────────────────────
e = a .^ 2;
assert(e(1,1,1) == 1)
assert(e(2,1,1) == 4)
assert(e(2,2,2) == 64)

% ── element-wise division ./ on 3D ───────────────────────────────
f = a ./ a;
assert(f(1,1,1) == 1)
assert(f(2,2,2) == 1)

% ── mod on 3D ─────────────────────────────────────────────────────
g = mod(a, 3);
assert(g(1,1,1) == 1)
assert(g(2,1,1) == 2)
assert(g(1,2,1) == 0)  % mod(3,3) = 0

% ── min/max of entire 3D tensor ───────────────────────────────────
assert(min(a(:)) == 1)
assert(max(a(:)) == 8)

disp('SUCCESS')
