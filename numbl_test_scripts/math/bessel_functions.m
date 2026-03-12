% Test Bessel functions: besselj, bessely, besseli, besselk
tol = 1e-10;

% ---- besselj: Bessel function of the first kind ----
% J_0(0) = 1
assert(abs(besselj(0, 0) - 1) < tol);
% J_1(0) = 0
assert(abs(besselj(1, 0)) < tol);
% J_0(1)
assert(abs(besselj(0, 1) - 0.7651976865579666) < tol);
% J_1(1)
assert(abs(besselj(1, 1) - 0.4400505857449335) < tol);
% J_0(5)
assert(abs(besselj(0, 5) - (-0.1775967713143383)) < tol);
% J_2(3)
assert(abs(besselj(2, 3) - 0.4860912605858911) < tol);
% Non-integer order: J_0.5(1)
assert(abs(besselj(0.5, 1) - 0.6713967071418030) < tol);

% ---- bessely: Bessel function of the second kind ----
% Y_0(1)
assert(abs(bessely(0, 1) - 0.0882569642156770) < tol);
% Y_1(1)
assert(abs(bessely(1, 1) - (-0.7812128213002887)) < tol);
% Y_0(5)
assert(abs(bessely(0, 5) - (-0.3085176252490338)) < tol);
% Y_2(3)
assert(abs(bessely(2, 3) - (-0.1604003934849238)) < tol);

% ---- besseli: Modified Bessel function of the first kind ----
% I_0(0) = 1
assert(abs(besseli(0, 0) - 1) < tol);
% I_1(0) = 0
assert(abs(besseli(1, 0)) < tol);
% I_0(1)
assert(abs(besseli(0, 1) - 1.2660658777520084) < tol);
% I_1(1)
assert(abs(besseli(1, 1) - 0.5651591039924851) < tol);

% ---- besselk: Modified Bessel function of the second kind ----
% K_0(1)
assert(abs(besselk(0, 1) - 0.4210244382407084) < tol);
% K_1(1)
assert(abs(besselk(1, 1) - 0.6019072301972346) < tol);
% K_0(0.5)
assert(abs(besselk(0, 0.5) - 0.9244190712276659) < tol);

% ---- Vector inputs ----
z = [1, 2, 3];
j0 = besselj(0, z);
assert(abs(j0(1) - 0.7651976865579666) < tol);
assert(abs(j0(2) - 0.2238907791412357) < tol);
assert(abs(j0(3) - (-0.2600519549019334)) < tol);

disp('SUCCESS');
