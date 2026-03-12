% Imaginary literals (1i, 2j) must not be shadowed by loop variables

% ── Basic imaginary literal ────────────────────────────────────────
z = 3i;
assert(abs(real(z)) < 1e-10)
assert(abs(imag(z) - 3) < 1e-10)

z2 = 1 + 2i;
assert(abs(real(z2) - 1) < 1e-10)
assert(abs(imag(z2) - 2) < 1e-10)

% ── After i is used as loop variable ──────────────────────────────
total = 0;
for i = 1:5
    total = total + i;
end
assert(total == 15)

% i is now 5 as a variable, but 1i must still be imaginary
z3 = 1 + 1i;
assert(abs(real(z3) - 1) < 1e-10)
assert(abs(imag(z3) - 1) < 1e-10)

z4 = [1+2i, 3+4i];
assert(abs(imag(z4(1)) - 2) < 1e-10)
assert(abs(imag(z4(2)) - 4) < 1e-10)

% ── j variant ─────────────────────────────────────────────────────
j = 10;
z5 = 2 + 3j;
assert(abs(real(z5) - 2) < 1e-10)
assert(abs(imag(z5) - 3) < 1e-10)

% ── Complex expressions after loop ────────────────────────────────
for i = 1:3
    for j = 1:3
    end
end
v = [1+1i, 2+2i, 3+3i];
assert(abs(imag(v(1)) - 1) < 1e-10)
assert(abs(imag(v(3)) - 3) < 1e-10)

% ── find on complex after loops ────────────────────────────────────
z6 = [1+1i, 2+0i, 0+3i, 4+4i];
big = find(abs(z6) > 2);
assert(length(big) == 2)
assert(big(1) == 3)
assert(big(2) == 4)

disp('SUCCESS')
