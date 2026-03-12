% Test airy function: Ai, Ai', Bi, Bi'

% Ai(x) at x=0
assert(abs(airy(0) - 0.355028053888) < 1e-10);

% Ai(x) at positive x
assert(abs(airy(1.5) - 0.071749497008) < 1e-10);

% Ai(x) at negative x
assert(abs(airy(-1) - 0.535560883292) < 1e-10);
assert(abs(airy(-5) - 0.350761009024) < 1e-8);

% airy(0, x) is same as airy(x)
assert(abs(airy(0, 1.5) - airy(1.5)) < 1e-15);

% Ai'(x) at x=0
assert(abs(airy(1, 0) - (-0.258819403793)) < 1e-10);

% Ai'(x) at positive and negative x
assert(abs(airy(1, 1) - (-0.159147441296793)) < 1e-10);
assert(abs(airy(1, -1) - (-0.010160567116645)) < 1e-8);

% Bi(x) at x=0
assert(abs(airy(2, 0) - 0.614926627446) < 1e-10);

% Bi(x) at positive and negative x
assert(abs(airy(2, 1) - 1.207423594952871) < 1e-10);
assert(abs(airy(2, -1) - 0.103997389496945) < 1e-8);

% Bi'(x) at x=0
assert(abs(airy(3, 0) - 0.448288357354) < 1e-10);

% Bi'(x) at positive and negative x
assert(abs(airy(3, 1) - 0.932435933392775) < 1e-10);
assert(abs(airy(3, -1) - 0.592375626422792) < 1e-8);

% Vector input
x = [-2, -1, 0, 1, 2];
y = airy(x);
assert(abs(y(3) - 0.355028053888) < 1e-10);
assert(abs(y(4) - airy(1)) < 1e-15);

% All four kinds at x=1
assert(abs(airy(0, 1) - airy(1)) < 1e-15);
assert(abs(airy(1, 1) - (-0.159147441296793)) < 1e-10);
assert(abs(airy(2, 1) - 1.207423594952871) < 1e-10);
assert(abs(airy(3, 1) - 0.932435933392775) < 1e-10);

% Scaled Ai: airy(0, x, 1) = exp(2/3 * x^(3/2)) * Ai(x) for x >= 0
zeta = (2/3) * 2^(3/2);
assert(abs(airy(0, 2, 1) - airy(0, 2) * exp(zeta)) < 1e-10);

% Scaled Bi: airy(2, x, 1) = exp(-2/3 * x^(3/2)) * Bi(x) for x >= 0
assert(abs(airy(2, 2, 1) - airy(2, 2) * exp(-zeta)) < 1e-10);

disp('SUCCESS');
