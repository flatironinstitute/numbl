% Basic arithmetic operations

assert(1 + 1 == 2)
assert(10 - 3 == 7)
assert(3 * 4 == 12)
assert(10 / 2 == 5)
assert(2 ^ 3 == 8)
assert(mod(10, 3) == 1)

% Floating point (32-bit precision, eps ~1.2e-7)
assert(abs(0.1 + 0.2 - 0.3) < 1e-6)

% Negative numbers
assert(-5 + 3 == -2)
assert(-4 * -3 == 12)

disp('SUCCESS')
