% Test: indexed read and assignment on a class property whose name could
% be confused with a method call.  Reproduces the chebfun trigtech bug:
%   f.values(:,f.isReal) = real(f.values(:,f.isReal));
% which raised "No method 'values' found".

f = MyVals([1+2i, 3+0i; 4+0i, 5+1i]);
f.isReal = [false, true];

% Read: f.values(:, mask) must resolve as property access + indexing
col = f.values(:, f.isReal);
assert(isequal(col, [3+0i; 5+1i]));

% Indexed assignment on property: f.values(:, mask) = expr
f.values(:, f.isReal) = real(f.values(:, f.isReal));
assert(isequal(f.values, [1+2i, 3; 4+0i, 5]));

disp('SUCCESS')
