% Test roots and poly builtins

% roots of a quadratic: x^2 - 5x + 6 = (x-2)(x-3)
r = roots([1 -5 6]);
r = sort(r);
assert(abs(r(1) - 2) < 1e-10);
assert(abs(r(2) - 3) < 1e-10);

% roots of x^2 - 1 = (x-1)(x+1)
r2 = roots([1 0 -1]);
r2 = sort(r2);
assert(abs(r2(1) - (-1)) < 1e-10);
assert(abs(r2(2) - 1) < 1e-10);

% roots of a cubic: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
r3 = roots([1 -6 11 -6]);
r3 = sort(r3);
assert(abs(r3(1) - 1) < 1e-10);
assert(abs(r3(2) - 2) < 1e-10);
assert(abs(r3(3) - 3) < 1e-10);

% Linear polynomial: 2x - 4 = 0 => x = 2
r4 = roots([2 -4]);
assert(abs(r4(1) - 2) < 1e-10);

% poly from roots: reconstruct polynomial from roots
p = poly([2 3]);
assert(abs(p(1) - 1) < 1e-10);
assert(abs(p(2) - (-5)) < 1e-10);
assert(abs(p(3) - 6) < 1e-10);

% poly from single root
p2 = poly([4]);
assert(abs(p2(1) - 1) < 1e-10);
assert(abs(p2(2) - (-4)) < 1e-10);

% Round-trip: roots -> poly -> roots
original_roots = [1 3 5];
p3 = poly(original_roots);
r5 = sort(roots(p3));
assert(norm(r5(:) - [1; 3; 5]) < 1e-10);

% poly from matrix returns characteristic polynomial
A = [1 2; 3 4];
cp = poly(A);
% Characteristic polynomial of [1 2; 3 4] is x^2 - 5x - 2
assert(abs(cp(1) - 1) < 1e-10);
assert(abs(cp(2) - (-5)) < 1e-10);
assert(abs(cp(3) - (-2)) < 1e-10);

disp('SUCCESS');
