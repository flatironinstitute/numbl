% Test round(x, n) - round to n decimal places

% Basic round(x, n) with positive n
assert(abs(round(3.14159, 2) - 3.14) < 1e-10);
assert(abs(round(3.145, 2) - 3.15) < 1e-10);
assert(abs(round(2.5, 0) - 3) < 1e-10);

% Negative n rounds to nearest 10^(-n)
assert(round(1234, -2) == 1200);
assert(round(1250, -2) == 1300);
assert(round(1249, -2) == 1200);

% Round array with n
r = round([1.23, 4.56, 7.89], 1);
assert(abs(r(1) - 1.2) < 1e-10);
assert(abs(r(2) - 4.6) < 1e-10);
assert(abs(r(3) - 7.9) < 1e-10);

% round(x) still works (1 arg)
assert(round(3.7) == 4);
assert(round(-3.7) == -4);
assert(round(0.5) == 1);
assert(round(-0.5) == -1);

disp('SUCCESS');
