% Test eps builtin function

% eps with no args returns 2^-52
assert(eps == 2^-52);
assert(eps() == 2^-52);

% eps(1.0) is equivalent to eps
assert(eps(1.0) == eps);

% eps('double') is equivalent to eps
assert(eps('double') == eps);

% eps('single') is equivalent to eps(single(1.0))
assert(eps('single') == 2^-23);

% log2(eps)
assert(log2(eps) == -52);

% eps(x) for specific values
assert(eps(1.0) == 2^-52);
assert(eps(0.5) == 2^-53);

% eps(10) should be ~1.7764e-15
assert(eps(10.0) > 0);
assert(eps(10.0) > eps(1.0));

% eps of negative values: eps(x) == eps(-x)
assert(eps(-1.0) == eps(1.0));
assert(eps(-10.0) == eps(10.0));

% eps(0) is the smallest positive subnormal
assert(eps(0) > 0);
assert(eps(0) < eps(1));

% eps(Inf) and eps(NaN) return NaN
assert(isnan(eps(Inf)));
assert(isnan(eps(NaN)));

% eps on a vector
v = eps([1 10 100]);
assert(length(v) == 3);
assert(v(1) == eps(1));
assert(v(2) == eps(10));
assert(v(3) == eps(100));

% eps on a matrix
M = eps([1 2; 3 4]);
assert(all(size(M) == [2 2]));
assert(M(1,1) == eps(1));
assert(M(1,2) == eps(2));
assert(M(2,1) == eps(3));
assert(M(2,2) == eps(4));

% larger values have larger eps
assert(eps(100) > eps(10));
assert(eps(10) > eps(1));
assert(eps(1) > eps(0.1));

disp('SUCCESS');
