% Test hyperbolic and degree-based trigonometric functions

% sinh
assert(abs(sinh(0)) < 1e-10, 'sinh(0)');
assert(abs(sinh(1) - 1.1752011936438) < 1e-10, 'sinh(1)');

% cosh
assert(abs(cosh(0) - 1) < 1e-10, 'cosh(0)');
assert(abs(cosh(1) - 1.5430806348152) < 1e-10, 'cosh(1)');

% tanh
assert(abs(tanh(0)) < 1e-10, 'tanh(0)');
assert(abs(tanh(0.5) - 0.46211715726001) < 1e-10, 'tanh(0.5)');

% sind (sine in degrees)
assert(abs(sind(0)) < 1e-10, 'sind(0)');
assert(abs(sind(30) - 0.5) < 1e-10, 'sind(30)');
assert(abs(sind(90) - 1) < 1e-10, 'sind(90)');
assert(abs(sind(180)) < 1e-10, 'sind(180)');

% cosd (cosine in degrees)
assert(abs(cosd(0) - 1) < 1e-10, 'cosd(0)');
assert(abs(cosd(60) - 0.5) < 1e-10, 'cosd(60)');
assert(abs(cosd(90)) < 1e-10, 'cosd(90)');

% tand (tangent in degrees)
assert(abs(tand(0)) < 1e-10, 'tand(0)');
assert(abs(tand(45) - 1) < 1e-10, 'tand(45)');

% Vector arguments
x = [0 1 2];
s = sinh(x);
assert(abs(s(1)) < 1e-10, 'sinh vector');
assert(abs(s(2) - sinh(1)) < 1e-10, 'sinh vector 2');

xd = [0 30 90];
sd = sind(xd);
assert(abs(sd(1)) < 1e-10, 'sind vector');
assert(abs(sd(2) - 0.5) < 1e-10, 'sind vector 2');
assert(abs(sd(3) - 1) < 1e-10, 'sind vector 3');

disp('SUCCESS');
