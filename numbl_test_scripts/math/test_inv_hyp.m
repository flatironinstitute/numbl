% Test inverse hyperbolic functions

% asinh
assert(abs(asinh(0)) < 1e-10, 'asinh(0)');
assert(abs(asinh(1) - 0.88137358701954) < 1e-10, 'asinh(1)');
assert(abs(asinh(-1) + 0.88137358701954) < 1e-10, 'asinh(-1)');

% acosh
assert(abs(acosh(1)) < 1e-10, 'acosh(1)');
assert(abs(acosh(2) - 1.31695789692482) < 1e-10, 'acosh(2)');
assert(abs(acosh(10) - 2.99322284612638) < 1e-10, 'acosh(10)');

% atanh
assert(abs(atanh(0)) < 1e-10, 'atanh(0)');
assert(abs(atanh(0.5) - 0.54930614433405) < 1e-10, 'atanh(0.5)');
assert(abs(atanh(-0.5) + 0.54930614433405) < 1e-10, 'atanh(-0.5)');

% asech
assert(abs(asech(1)) < 1e-10, 'asech(1)');
assert(abs(asech(0.5) - 1.31695789692482) < 1e-10, 'asech(0.5)');

% acsch
assert(abs(acsch(1) - 0.88137358701954) < 1e-10, 'acsch(1)');
assert(abs(acsch(2) - 0.48121182505960) < 1e-10, 'acsch(2)');

% acoth
assert(abs(acoth(2) - 0.54930614433405) < 1e-10, 'acoth(2)');
assert(abs(acoth(10) - 0.10033534773108) < 1e-10, 'acoth(10)');

% Vector arguments
x = [1 2 3];
s = asinh(x);
assert(abs(s(1) - asinh(1)) < 1e-10, 'asinh vector');
assert(abs(s(2) - asinh(2)) < 1e-10, 'asinh vector 2');

c = acosh(x);
assert(abs(c(1) - acosh(1)) < 1e-10, 'acosh vector');
assert(abs(c(2) - acosh(2)) < 1e-10, 'acosh vector 2');

disp('SUCCESS');
