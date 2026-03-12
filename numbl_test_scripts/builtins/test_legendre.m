% Test legendre builtin

% Basic: P_0(x) = 1
P = legendre(0, 0.5);
assert(abs(P - 1) < 1e-10, 'P_0(0.5) should be 1');

% P_1(x): m=0 gives x, m=1 gives -sqrt(1-x^2)
P = legendre(1, 0.5);
assert(abs(P(1) - 0.5) < 1e-10, 'P_1^0(0.5) should be 0.5');
assert(abs(P(2) - (-sqrt(1 - 0.25))) < 1e-10, 'P_1^1(0.5)');

% P_2(x) at x=0: from the MATLAB docs
P = legendre(2, 0);
assert(abs(P(1) - (-0.5)) < 1e-10, 'P_2^0(0) = -0.5');
assert(abs(P(2) - 0) < 1e-10, 'P_2^1(0) = 0');
assert(abs(P(3) - 3) < 1e-10, 'P_2^2(0) = 3');

% Vector input: legendre(2, [0, 0.1, 0.2])
deg = 2;
x = [0, 0.1, 0.2];
P = legendre(deg, x);
assert(size(P, 1) == 3, 'Should have 3 rows (m=0,1,2)');
assert(size(P, 2) == 3, 'Should have 3 columns');
% Check specific values from MATLAB docs
assert(abs(P(1,1) - (-0.5)) < 1e-4, 'P_2^0(0) = -0.5');
assert(abs(P(1,2) - (-0.485)) < 1e-3, 'P_2^0(0.1)');
assert(abs(P(1,3) - (-0.44)) < 1e-3, 'P_2^0(0.2)');
assert(abs(P(2,1) - 0) < 1e-10, 'P_2^1(0) = 0');
assert(abs(P(3,1) - 3) < 1e-4, 'P_2^2(0) = 3');
assert(abs(P(3,2) - 2.97) < 1e-2, 'P_2^2(0.1)');
assert(abs(P(3,3) - 2.88) < 1e-2, 'P_2^2(0.2)');

% Schmidt seminormalized: legendre(1, x, 'sch')
x = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
P_sch = legendre(1, x, 'sch');
% m=0 row should be same as unnormalized
assert(abs(P_sch(1,1) - 0) < 1e-10, 'sch m=0 at x=0');
assert(abs(P_sch(1,2) - 0.2) < 1e-10, 'sch m=0 at x=0.2');
% m=1 row: scaling is (-1)*sqrt(2*0!/2!) = -1 times unnorm
% unnorm P_1^1(0) = -1, so sch = (-1)*(-1) = 1
assert(abs(P_sch(2,1) - 1) < 1e-10, 'sch m=1 at x=0');
assert(abs(P_sch(2,6) - 0) < 1e-10, 'sch m=1 at x=1');

% Fully normalized: legendre(1, x, 'norm')
P_norm = legendre(1, x, 'norm');
% m=0: scale = sqrt(3/2) * unnorm
% unnorm P_1^0(0) = 0
assert(abs(P_norm(1,1) - 0) < 1e-10, 'norm m=0 at x=0');
% m=0 at x=1: unnorm = 1, scale = sqrt(1.5) = 1.2247
assert(abs(P_norm(1,6) - sqrt(1.5)) < 1e-4, 'norm m=0 at x=1');
% m=1 at x=0: unnorm = -1, scale = (-1)*sqrt(1.5/2) = -0.8660
% result = -0.8660 * (-1) = 0.8660
assert(abs(P_norm(2,1) - 0.8660) < 1e-3, 'norm m=1 at x=0');

% Higher degree: legendre(3, 0)
P = legendre(3, 0);
assert(size(P, 1) == 4, 'Should have 4 rows for degree 3');
% P_3^0(0) = 0
assert(abs(P(1)) < 1e-10, 'P_3^0(0) = 0');

% Test at x=1 (boundary)
P = legendre(2, 1);
assert(abs(P(1) - 1) < 1e-10, 'P_2^0(1) = 1');
assert(abs(P(2) - 0) < 1e-10, 'P_2^1(1) = 0');
assert(abs(P(3) - 0) < 1e-10, 'P_2^2(1) = 0');

% Test at x=-1 (boundary)
P = legendre(2, -1);
assert(abs(P(1) - 1) < 1e-10, 'P_2^0(-1) = 1');
assert(abs(P(2) - 0) < 1e-10, 'P_2^1(-1) = 0');
assert(abs(P(3) - 0) < 1e-10, 'P_2^2(-1) = 0');

disp('SUCCESS');
