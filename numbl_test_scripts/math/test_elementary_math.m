% Test elementary math functions: hypot, nthroot, log1p, expm1, pow2, nextpow2

%% hypot
assert(hypot(3, 4) == 5);
assert(hypot(0, 0) == 0);
assert(hypot(1, 0) == 1);
assert(abs(hypot(1, 1) - sqrt(2)) < 1e-10);

% hypot on vectors
v = hypot([3, 5, 8], [4, 12, 15]);
assert(v(1) == 5);
assert(v(2) == 13);
assert(v(3) == 17);

% hypot with scalar expansion
v = hypot([3, 5], 4);
assert(v(1) == 5);
assert(abs(v(2) - sqrt(41)) < 1e-10);

%% nthroot
assert(nthroot(27, 3) == 3);
assert(nthroot(8, 3) == 2);
assert(nthroot(-8, 3) == -2);
assert(nthroot(-27, 3) == -3);
assert(nthroot(16, 4) == 2);
assert(nthroot(1, 5) == 1);
assert(abs(nthroot(2, 2) - sqrt(2)) < 1e-10);

% nthroot on vectors
v = nthroot([8, 27, 64], 3);
assert(v(1) == 2);
assert(v(2) == 3);
assert(v(3) == 4);

%% log1p
assert(log1p(0) == 0);
assert(abs(log1p(1) - log(2)) < 1e-10);
assert(abs(log1p(exp(1) - 1) - 1) < 1e-10);

% log1p on vectors
v = log1p([0, 1, exp(1)-1]);
assert(abs(v(1) - 0) < 1e-10);
assert(abs(v(2) - log(2)) < 1e-10);
assert(abs(v(3) - 1) < 1e-10);

%% expm1
assert(expm1(0) == 0);
assert(abs(expm1(1) - (exp(1) - 1)) < 1e-10);
assert(abs(expm1(log(2)) - 1) < 1e-10);

% expm1 on vectors
v = expm1([0, 1, log(2)]);
assert(abs(v(1) - 0) < 1e-10);
assert(abs(v(2) - (exp(1) - 1)) < 1e-10);
assert(abs(v(3) - 1) < 1e-10);

%% pow2
assert(pow2(0) == 1);
assert(pow2(1) == 2);
assert(pow2(2) == 4);
assert(pow2(3) == 8);
assert(pow2(-1) == 0.5);
assert(pow2(10) == 1024);

% pow2 on vectors
v = pow2([0, 1, 2, 3, 10]);
assert(v(1) == 1);
assert(v(2) == 2);
assert(v(3) == 4);
assert(v(4) == 8);
assert(v(5) == 1024);

%% nextpow2
assert(nextpow2(1) == 0);
assert(nextpow2(2) == 1);
assert(nextpow2(3) == 2);
assert(nextpow2(4) == 2);
assert(nextpow2(5) == 3);
assert(nextpow2(8) == 3);
assert(nextpow2(9) == 4);
assert(nextpow2(1024) == 10);
assert(nextpow2(1025) == 11);

% nextpow2 on vectors
v = nextpow2([1, 2, 3, 4, 5]);
assert(v(1) == 0);
assert(v(2) == 1);
assert(v(3) == 2);
assert(v(4) == 2);
assert(v(5) == 3);

disp('SUCCESS');
