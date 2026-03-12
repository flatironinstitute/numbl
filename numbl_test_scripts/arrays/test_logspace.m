% Test logspace builtin

%% Basic usage
v = logspace(1, 3, 3);
assert(length(v) == 3);
assert(abs(v(1) - 10) < 1e-10);
assert(abs(v(2) - 100) < 1e-10);
assert(abs(v(3) - 1000) < 1e-10);

%% Default n=50
v = logspace(0, 1);
assert(length(v) == 50);
assert(abs(v(1) - 1) < 1e-10);
assert(abs(v(end) - 10) < 1e-10);

%% Single point
v = logspace(2, 2, 1);
assert(length(v) == 1);
assert(abs(v(1) - 100) < 1e-10);

%% Two points
v = logspace(0, 2, 2);
assert(abs(v(1) - 1) < 1e-10);
assert(abs(v(2) - 100) < 1e-10);

%% Negative exponents
v = logspace(-1, 1, 3);
assert(abs(v(1) - 0.1) < 1e-12);
assert(abs(v(2) - 1) < 1e-10);
assert(abs(v(3) - 10) < 1e-10);

%% Special case: b == pi
v = logspace(0, 1, 3);
assert(abs(v(1) - 1) < 1e-10);

%% logspace with pi as endpoint
v = logspace(0, pi, 4);
assert(abs(v(1) - 1) < 1e-10);
assert(abs(v(end) - pi) < 1e-10);

%% Output is a row vector
v = logspace(1, 2, 5);
assert(size(v, 1) == 1);
assert(size(v, 2) == 5);

disp('SUCCESS');
