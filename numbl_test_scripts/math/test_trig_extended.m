% Test degree-based inverse trig and reciprocal trig functions

%% asind, acosd, atand
assert(asind(0) == 0);
assert(asind(1) == 90);
assert(asind(-1) == -90);
assert(abs(asind(0.5) - 30) < 1e-10);

assert(acosd(1) == 0);
assert(acosd(0) == 90);
assert(acosd(-1) == 180);
assert(abs(acosd(0.5) - 60) < 1e-10);

assert(atand(0) == 0);
assert(atand(1) == 45);
assert(atand(-1) == -45);
assert(abs(atand(sqrt(3)) - 60) < 1e-10);

%% atan2d
assert(atan2d(0, 1) == 0);
assert(atan2d(1, 0) == 90);
assert(atan2d(0, -1) == 180);
assert(atan2d(-1, 0) == -90);
assert(atan2d(1, 1) == 45);

%% asind/acosd/atand on vectors
v = asind([0, 0.5, 1]);
assert(abs(v(1) - 0) < 1e-10);
assert(abs(v(2) - 30) < 1e-10);
assert(abs(v(3) - 90) < 1e-10);

%% sec, csc, cot
assert(abs(sec(0) - 1) < 1e-10);
assert(abs(csc(pi/2) - 1) < 1e-10);
assert(abs(cot(pi/4) - 1) < 1e-10);

% sec = 1/cos
x = 0.7;
assert(abs(sec(x) - 1/cos(x)) < 1e-10);
assert(abs(csc(x) - 1/sin(x)) < 1e-10);
assert(abs(cot(x) - cos(x)/sin(x)) < 1e-10);

% on vectors
v = sec([0, pi/3, pi/4]);
assert(abs(v(1) - 1) < 1e-10);
assert(abs(v(2) - 2) < 1e-10);
assert(abs(v(3) - sqrt(2)) < 1e-10);

%% sech, csch, coth
assert(abs(sech(0) - 1) < 1e-10);
x = 1.0;
assert(abs(sech(x) - 1/cosh(x)) < 1e-10);
assert(abs(csch(x) - 1/sinh(x)) < 1e-10);
assert(abs(coth(x) - cosh(x)/sinh(x)) < 1e-10);

%% secd, cscd, cotd
assert(abs(secd(0) - 1) < 1e-10);
assert(abs(secd(60) - 2) < 1e-10);
assert(abs(cscd(90) - 1) < 1e-10);
assert(abs(cscd(30) - 2) < 1e-10);
assert(abs(cotd(45) - 1) < 1e-10);

%% asec, acsc, acot
assert(abs(asec(1) - 0) < 1e-10);
assert(abs(asec(2) - pi/3) < 1e-10);
assert(abs(acsc(1) - pi/2) < 1e-10);
assert(abs(acsc(2) - pi/6) < 1e-10);
assert(abs(acot(1) - pi/4) < 1e-10);
assert(abs(acot(0) - pi/2) < 1e-10);

%% asecd, acscd, acotd
assert(abs(asecd(1) - 0) < 1e-10);
assert(abs(asecd(2) - 60) < 1e-10);
assert(abs(acscd(1) - 90) < 1e-10);
assert(abs(acscd(2) - 30) < 1e-10);
assert(abs(acotd(1) - 45) < 1e-10);
assert(abs(acotd(0) - 90) < 1e-10);

disp('SUCCESS');
