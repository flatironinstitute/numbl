% Test rounding and modulo functions

% floor
assert(floor(3.7) == 3);
assert(floor(-3.7) == -4);
assert(floor(3.0) == 3);

% ceil
assert(ceil(3.2) == 4);
assert(ceil(-3.2) == -3);
assert(ceil(3.0) == 3);

% round
assert(round(3.4) == 3);
assert(round(3.5) == 4);
assert(round(-3.5) == -4);

% fix (truncate toward zero)
assert(fix(3.7) == 3);
assert(fix(-3.7) == -3);

% mod (same sign as divisor)
assert(mod(10, 3) == 1);
assert(mod(-10, 3) == 2);
assert(mod(10, -3) == -2);

% rem (same sign as dividend)
assert(rem(10, 3) == 1);
assert(rem(-10, 3) == -1);

% abs
assert(abs(-5) == 5);
assert(abs(5) == 5);
assert(abs(0) == 0);

% sign
assert(sign(5) == 1);
assert(sign(-5) == -1);
assert(sign(0) == 0);

disp('SUCCESS')
