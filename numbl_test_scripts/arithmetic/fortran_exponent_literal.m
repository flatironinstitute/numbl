% FORTRAN-style double-precision exponent literals (d/D) should parse
% identically to e/E exponent markers. MATLAB accepts both.

% ── Basic forms ────────────────────────────────────────────────────
assert(2.0d0 == 2.0)
assert(2.0D0 == 2.0)
assert(1d2 == 100)
assert(1D2 == 100)
assert(1.5d3 == 1500)

% ── Signed exponents ───────────────────────────────────────────────
assert(1d-2 == 0.01)
assert(1d+3 == 1000)
assert(abs(2.5d-1 - 0.25) < 1e-12)

% ── In expressions ─────────────────────────────────────────────────
x = 3.0d0 * 4.0d0;
assert(x == 12.0)

y = (1.0d0 + 2.0d0) / 3.0d0;
assert(abs(y - 1.0) < 1e-12)

% ── Mixed d/D and e/E in the same expression ───────────────────────
assert(1d3 == 1e3)
assert(2.5D-2 == 2.5e-2)

% ── In an array ────────────────────────────────────────────────────
v = [1.0d0, 2.0d0, 3.0d0];
assert(length(v) == 3)
assert(v(2) == 2.0)

% ── Matches the usage that triggered the bug (bymode.m) ────────────
a = [1 2 3];
b = a * 2.0d0;
assert(b(1) == 2)
assert(b(3) == 6)

disp('SUCCESS')
