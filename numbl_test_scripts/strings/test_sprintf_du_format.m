% Test sprintf %d with non-integer floats and %u format specifier

% %d with integer values works normally
assert(strcmp(sprintf('%d', 42), '42'));
assert(strcmp(sprintf('%d', 0), '0'));
assert(strcmp(sprintf('%d', -5), '-5'));

% %d with non-integer floats: MATLAB falls back to scientific notation
assert(strcmp(sprintf('%d', 3.7), '3.700000e+00'));
assert(strcmp(sprintf('%d', -2.5), '-2.500000e+00'));
assert(strcmp(sprintf('%d', 0.001), '1.000000e-03'));

% %u with positive integer
assert(strcmp(sprintf('%u', 42), '42'));
assert(strcmp(sprintf('%u', 0), '0'));

% %u with non-integer: falls back to scientific notation
assert(strcmp(sprintf('%u', 3.7), '3.700000e+00'));

% %u with negative: falls back to scientific notation
assert(strcmp(sprintf('%u', -1), '-1.000000e+00'));

disp('SUCCESS');
