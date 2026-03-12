% Anonymous functions

sq = @(x) x^2;
assert(sq(3) == 9)
assert(sq(5) == 25)

add = @(a, b) a + b;
assert(add(3, 4) == 7)

% Composition
double_then_square = @(x) sq(x * 2);
assert(double_then_square(3) == 36)

% Closures capture variables
k = 10;
add_k = @(x) x + k;
assert(add_k(5) == 15)

% Function handles to builtins
f = @sin;
assert(abs(f(0)) < 1e-5)
assert(abs(f(pi/2) - 1) < 1e-5)

% arrayfun
v = [1, 2, 3, 4];
v2 = arrayfun(@(x) x^2, v);
assert(v2(1) == 1)
assert(v2(2) == 4)
assert(v2(3) == 9)
assert(v2(4) == 16)

disp('SUCCESS')
