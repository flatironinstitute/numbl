% Test that element-wise functions on negative real tensors produce complex results

% sqrt of negative vector
sq = sqrt([-1, -4, -9]);
assert(abs(imag(sq(1)) - 1) < 1e-10, 'sqrt(-1) tensor');
assert(abs(imag(sq(2)) - 2) < 1e-10, 'sqrt(-4) tensor');
assert(abs(imag(sq(3)) - 3) < 1e-10, 'sqrt(-9) tensor');
assert(abs(real(sq(1))) < 1e-10, 'sqrt(-1) real part should be 0');
assert(abs(real(sq(2))) < 1e-10, 'sqrt(-4) real part should be 0');

% sqrt of negative matrix
M = [-1 -4; -9 -16];
sqM = sqrt(M);
assert(abs(imag(sqM(1,1)) - 1) < 1e-10, 'sqrt matrix (1,1)');
assert(abs(imag(sqM(2,2)) - 4) < 1e-10, 'sqrt matrix (2,2)');

% sqrt of mixed positive and negative
mixed = [4, -4, 9, -9];
sqMixed = sqrt(mixed);
assert(abs(sqMixed(1) - 2) < 1e-10, 'sqrt(4)');
assert(abs(imag(sqMixed(2)) - 2) < 1e-10, 'sqrt(-4) mixed');
assert(abs(sqMixed(3) - 3) < 1e-10, 'sqrt(9)');
assert(abs(imag(sqMixed(4)) - 3) < 1e-10, 'sqrt(-9) mixed');

% power of negative tensor with fractional exponent
p = [-1, -8].^(1/3);
p1_re = real(p(1));
p1_im = imag(p(1));
assert(abs(sqrt(p1_re^2 + p1_im^2) - 1) < 1e-10, 'power (-1)^(1/3) magnitude');
assert(p1_im ~= 0, 'power (-1)^(1/3) should be complex');

disp('SUCCESS');
