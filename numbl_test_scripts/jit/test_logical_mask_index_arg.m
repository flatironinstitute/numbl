% A logical mask passed as an argument into a JIT-compiled function and
% used as a per-axis index must select by the mask, not gather by numeric
% value. Regression: the JIT type adapter dropped the logical flag, so
% M(mask, :) took the numeric-gather path and returned the wrong rows.
%
% The displayed (unsuppressed) assignments route the calls through the
% interpreter -> JIT call boundary, which is where the dropped flag bit.

M = [1 2; 3 4; 5 6];

% all-true mask selects every row
maskAll = logical([1 1 1]);
Rall = f(M, maskAll)
assert(isequal(Rall, M));

% partial mask selects a subset, in order
maskPart = logical([1 0 1]);
Rpart = f(M, maskPart)
assert(isequal(Rpart, [1 2; 5 6]));

% column mask
maskCol = logical([0 1]);
Ccol = g(M, maskCol)
assert(isequal(Ccol, [2; 4; 6]));

disp('SUCCESS')

function R = f(M, rowmask)
  R = M(rowmask, :);
end

function C = g(M, colmask)
  C = M(:, colmask);
end
