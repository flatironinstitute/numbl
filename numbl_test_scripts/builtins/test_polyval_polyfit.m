% Test polyval and polyfit builtins

% polyval: evaluate polynomial
% p(x) = 2x^2 + 3x + 1
p = [2 3 1];
assert(polyval(p, 0) == 1);
assert(polyval(p, 1) == 6);
assert(polyval(p, 2) == 15);
assert(polyval(p, -1) == 0);

% polyval with vector input
x = [0 1 2 3];
y = polyval(p, x);
assert(isequal(y, [1 6 15 28]));

% polyfit: fit polynomial to data
x2 = [1 2 3 4 5];
y2 = 2*x2.^2 + 3*x2 + 1;
p2 = polyfit(x2, y2, 2);
assert(norm(p2 - [2 3 1]) < 1e-8);

% Linear fit
x3 = [0 1 2 3 4];
y3 = 2*x3 + 5;
p3 = polyfit(x3, y3, 1);
assert(abs(p3(1) - 2) < 1e-10);
assert(abs(p3(2) - 5) < 1e-10);

% Constant fit
x4 = [1 2 3];
y4 = [7 7 7];
p4 = polyfit(x4, y4, 0);
assert(abs(p4 - 7) < 1e-10);

disp('SUCCESS');
