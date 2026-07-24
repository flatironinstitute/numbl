% normalize: default zscore, and the 'norm' method (p-norm scaling).

% 'norm' on a column vector: divide by 2-norm
v = normalize([3; 4], 'norm');
assert(max(abs(v - [0.6; 0.8])) < 1e-12);

% Column-wise for matrices
M = [3 0; 4 2];
N = normalize(M, 'norm');
assert(max(abs(N(:, 1) - [0.6; 0.8])) < 1e-12);
assert(max(abs(N(:, 2) - [0; 1])) < 1e-12);

% Row vector normalizes along dim 2
r = normalize([3 4], 'norm');
assert(max(abs(r - [0.6 0.8])) < 1e-12);

% Explicit p-norm
v1 = normalize([1; 1], 'norm', 1);
assert(max(abs(v1 - [0.5; 0.5])) < 1e-12);
vinf = normalize([2; -4], 'norm', Inf);
assert(max(abs(vinf - [0.5; -1])) < 1e-12);

% Default method is zscore
z = normalize([1 2 3]);
assert(max(abs(z - [-1 0 1])) < 1e-12);
z2 = normalize([1; 2; 3]);
assert(max(abs(z2 - [-1; 0; 1])) < 1e-12);

disp('SUCCESS');
