% Test that bundled shim files are available via search path resolution

% --- decomposition shim (classdef at shims root) ---
A = [2 1; 5 3];
dA = decomposition(A, 'CheckCondition', false);
x = dA \ [4; 11];
assert(abs(x(1) - 1) < 1e-10);
assert(abs(x(2) - 2) < 1e-10);

% isIllConditioned shim (function at shims root)
assert(~isIllConditioned(dA));

% --- +matlab/+internal/+decomposition/DenseLU (package shim) ---
lu_obj = matlab.internal.decomposition.DenseLU(A);
x2 = lu_obj.solve([4; 11], false);
assert(abs(x2(1) - 1) < 1e-10);
assert(abs(x2(2) - 2) < 1e-10);

% --- +matlab/+internal/+math/blkdiag (package shim) ---
B = matlab.internal.math.blkdiag([1 2], [3]);
assert(isequal(B, [1 2 0; 0 0 3]));

disp('SUCCESS');
