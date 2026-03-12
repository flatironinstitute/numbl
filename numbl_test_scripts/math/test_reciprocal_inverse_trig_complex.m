% Test reciprocal inverse trig/hyperbolic functions returning complex values
% These should return complex results for out-of-domain real inputs

tol = 1e-10;

%% acoth - should return complex for |x| < 1
r = acoth(0.5);
assert(abs(real(r) - 0.5493061443340549) < tol);
assert(abs(imag(r) - pi/2) < tol);

r = acoth(-0.5);
assert(abs(real(r) - (-0.5493061443340549)) < tol);
assert(abs(imag(r) - (-pi/2)) < tol);

% acoth for |x| > 1 should be real
r = acoth(2);
assert(abs(r - 0.5493061443340549) < tol);
assert(imag(r) == 0);

%% asec - should return complex for |x| < 1
r = asec(0.5);
assert(abs(real(r) - 0) < tol);
assert(abs(imag(r) - 1.3169578969248166) < tol);

r = asec(-0.5);
assert(abs(real(r) - pi) < tol);
assert(abs(imag(r) - (-1.3169578969248166)) < tol);

% asec for |x| >= 1 should be real
r = asec(2);
assert(abs(r - acos(0.5)) < tol);
assert(imag(r) == 0);

%% acsc - should return complex for |x| < 1
r = acsc(0.5);
assert(abs(real(r) - pi/2) < tol);
assert(abs(imag(r) - (-1.3169578969248166)) < tol);

r = acsc(-0.5);
assert(abs(real(r) - (-pi/2)) < tol);
assert(abs(imag(r) - 1.3169578969248166) < tol);

% acsc for |x| >= 1 should be real
r = acsc(2);
assert(abs(r - asin(0.5)) < tol);
assert(imag(r) == 0);

%% asech - should return complex for x > 1 or x < 0
r = asech(2);
assert(abs(real(r) - 0) < tol);
assert(abs(imag(r) - pi/3) < tol);

r = asech(-0.5);
assert(abs(real(r) - 1.3169578969248166) < tol);
assert(abs(imag(r) - pi) < tol);

% asech for 0 < x <= 1 should be real
r = asech(0.5);
assert(abs(r - acosh(2)) < tol);
assert(imag(r) == 0);

%% Complex inputs should also work
r = acoth(1i);
expected = atanh(1/(1i));
assert(abs(r - expected) < tol);

r = asec(1i);
expected = acos(1/(1i));
assert(abs(r - expected) < tol);

%% Tensor inputs
v = [0.5, 2, -0.5];
r = acoth(v);
assert(abs(real(r(1)) - 0.5493061443340549) < tol);
assert(abs(imag(r(1)) - pi/2) < tol);
assert(abs(r(2) - 0.5493061443340549) < tol);
assert(imag(r(2)) == 0);

disp('SUCCESS');
