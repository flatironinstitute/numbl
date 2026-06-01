% Hyperbolic sinh / cosh / tanh under the JIT (JS- and C-JIT).
% Exercises real scalars, real tensors, and complex scalars.

%!numbl:assert_jit c

% Real scalars: reference values
assert(abs(sinh(0)) < 1e-12);
assert(abs(cosh(0) - 1) < 1e-12);
assert(abs(tanh(0)) < 1e-12);
assert(abs(sinh(1) - 1.1752011936438014) < 1e-12);
assert(abs(cosh(1) - 1.5430806348152437) < 1e-12);
assert(abs(tanh(0.5) - 0.46211715726000974) < 1e-12);

% Identities
x = 1.3;
assert(abs(cosh(x) * cosh(x) - sinh(x) * sinh(x) - 1) < 1e-12);
assert(abs(tanh(x) - sinh(x) / cosh(x)) < 1e-12);

% Sign behaviour (odd functions)
assert(sinh(-2) == -sinh(2));
assert(tanh(-2) == -tanh(2));
assert(cosh(-2) == cosh(2));

% Real tensors
v = [0 1 2 3];
s = sinh(v);
c = cosh(v);
t = tanh(v);
assert(abs(s(1)) < 1e-12);
assert(abs(s(2) - sinh(1)) < 1e-12);
assert(abs(c(3) - cosh(2)) < 1e-12);
assert(abs(t(4) - tanh(3)) < 1e-12);
% cosh^2 - sinh^2 == 1 elementwise
d = c .* c - s .* s;
assert(max(abs(d - 1)) < 1e-12);

% Complex scalars via identities:
%   sinh(i*y) = i*sin(y), cosh(i*y) = cos(y), tanh(i*y) = i*tan(y)
zs = sinh(2i);
assert(abs(real(zs)) < 1e-12);
assert(abs(imag(zs) - sin(2)) < 1e-12);

zc = cosh(2i);
assert(abs(real(zc) - cos(2)) < 1e-12);
assert(abs(imag(zc)) < 1e-12);

zt = tanh(1i);
assert(abs(real(zt)) < 1e-12);
assert(abs(imag(zt) - tan(1)) < 1e-12);

% General complex value against the componentwise formula
w = sinh(1.0 + 0.5i);
assert(abs(real(w) - sinh(1.0) * cos(0.5)) < 1e-12);
assert(abs(imag(w) - cosh(1.0) * sin(0.5)) < 1e-12);

% asinh (real-only): inverse of sinh, odd, entire on the reals
assert(abs(asinh(0)) < 1e-12);
assert(abs(asinh(1) - 0.881373587019543) < 1e-12);
assert(asinh(-3) == -asinh(3));
assert(abs(sinh(asinh(2.5)) - 2.5) < 1e-12);
a = asinh([0 1 2 3]);
assert(abs(a(2) - asinh(1)) < 1e-12);
assert(max(abs(sinh(a) - [0 1 2 3])) < 1e-12);

disp('SUCCESS');
