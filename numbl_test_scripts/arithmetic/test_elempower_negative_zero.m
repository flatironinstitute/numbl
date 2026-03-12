% Test element-wise power (.^) on tensors with negative values, zeros,
% and integer exponents.
%
% MATLAB behavior:
% - (-x).^2 should be real (not complex) for integer exponents
% - 0.^2 should be 0 (not NaN), even if other elements are negative
% - (-x).^(1/3) produces complex (this is already tested elsewhere)

% Test 1: vector with negatives raised to integer power
x = [-2, -1, 0, 1, 2];
r = x.^2;
assert(r(1) == 4, '(-2).^2 should be 4');
assert(r(2) == 1, '(-1).^2 should be 1');
assert(r(3) == 0, '0.^2 should be 0');
assert(r(4) == 1, '1.^2 should be 1');
assert(r(5) == 4, '2.^2 should be 4');
assert(isreal(r), 'integer power of real vector should be real');

% Test 2: vector with negatives raised to odd integer power
r2 = x.^3;
assert(r2(1) == -8, '(-2).^3 should be -8');
assert(r2(2) == -1, '(-1).^3 should be -1');
assert(r2(3) == 0, '0.^3 should be 0');
assert(r2(4) == 1, '1.^3 should be 1');
assert(r2(5) == 8, '2.^3 should be 8');
assert(isreal(r2), 'integer power of real vector should be real');

% Test 3: zero in vector raised to power should never be NaN
y = [-1, 0, 1];
r3 = y.^2;
assert(~any(isnan(r3)), 'no NaN in y.^2');
assert(r3(2) == 0, '0.^2 should be 0 in mixed vector');
assert(isreal(r3), 'y.^2 should be real for integer exponent');

% Test 4: 0.^0 should be 1 (MATLAB convention)
r4 = [0, 1, 2].^0;
assert(r4(1) == 1, '0.^0 should be 1');
assert(r4(2) == 1, '1.^0 should be 1');

% Test 5: negative values raised to power 1
r5 = [-3, -2, -1, 0, 1].^1;
assert(r5(1) == -3, '(-3).^1 should be -3');
assert(r5(4) == 0, '0.^1 should be 0');
assert(isreal(r5), '.^1 should be real');

% Test 6: 1-x.^2 pattern (common in Legendre polynomial code)
z = [-0.7746, 0, 0.7746];
r6 = 1 - z.^2;
assert(~any(isnan(r6)), 'no NaN in 1-z.^2');
assert(isreal(r6), '1-z.^2 should be real');
assert(abs(r6(2) - 1) < 1e-10, '1 - 0^2 should be 1');
assert(abs(r6(1) - r6(3)) < 1e-10, 'symmetric values should give same result');

% Test 7: matrix with mixed values
M = [-1 2; 0 -3];
r7 = M.^2;
assert(r7(1,1) == 1, '(-1)^2 should be 1');
assert(r7(1,2) == 4, '2^2 should be 4');
assert(r7(2,1) == 0, '0^2 should be 0');
assert(r7(2,2) == 9, '(-3)^2 should be 9');
assert(isreal(r7), 'integer power of real matrix should be real');

% Test 8: negative values with fractional exponent produce complex (keep existing behavior)
r8 = [-1, -4].^0.5;
assert(~isreal(r8), 'fractional power of negatives should be complex');
assert(abs(imag(r8(1)) - 1) < 1e-10, 'sqrt(-1) should give 1i');

% Test 9: mixed vector with fractional exponent - zeros should still work
r9 = [-1, 0, 1].^0.5;
assert(~isnan(real(r9(2))), '0.^0.5 should not be NaN');
assert(abs(r9(2)) < 1e-10, '0.^0.5 should be 0');
assert(abs(r9(3) - 1) < 1e-10, '1.^0.5 should be 1');

disp('SUCCESS');
