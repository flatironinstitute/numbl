% Math builtin functions (32-bit precision, tol ~1e-5)
tol = 1e-5;

% Trig
assert(abs(sin(0)) < tol)
assert(abs(sin(pi/2) - 1) < tol)
assert(abs(cos(0) - 1) < tol)
assert(abs(cos(pi) + 1) < tol)
assert(abs(tan(pi/4) - 1) < tol)

% Inverse trig
assert(abs(asin(1) - pi/2) < tol)
assert(abs(acos(1)) < tol)
assert(abs(atan(1) - pi/4) < tol)
assert(abs(atan2(1, 1) - pi/4) < tol)

% Exponential / log
assert(abs(exp(0) - 1) < tol)
assert(abs(exp(1) - exp(1)) < tol)
assert(abs(log(1)) < tol)
assert(abs(log(exp(3)) - 3) < tol)
assert(abs(log10(100) - 2) < tol)
assert(abs(log2(8) - 3) < tol)

% Powers / roots
assert(abs(sqrt(4) - 2) < tol)
assert(abs(sqrt(9) - 3) < tol)

% Rounding
assert(floor(3.7) == 3)
assert(ceil(3.2) == 4)
assert(round(3.5) == 4)
assert(round(3.4) == 3)
assert(fix(3.9) == 3)
assert(fix(-3.9) == -3)

% abs / sign
assert(abs(-5) == 5)
assert(abs(5) == 5)
assert(sign(-3) == -1)
assert(sign(0) == 0)
assert(sign(4) == 1)

% factorial
assert(factorial(0) == 1)
assert(factorial(1) == 1)
assert(factorial(5) == 120)
assert(factorial(10) == 3628800)

disp('SUCCESS')
