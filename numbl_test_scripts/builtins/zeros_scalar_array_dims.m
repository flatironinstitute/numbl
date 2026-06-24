% Size constructors (zeros/ones/...) accept a 1x1 array as a scalar dimension
% in multi-argument form, matching MATLAB. A 1x1 array counts as a scalar;
% a multi-element vector is only valid as the sole argument.

n = [5];        % a 1x1 array, not a literal scalar
m = 4;

a = zeros(n, m);
assert(isequal(size(a), [5 4]), 'zeros(1x1, scalar) shape');
assert(all(a(:) == 0), 'zeros values');

b = ones(m, n);
assert(isequal(size(b), [4 5]), 'ones(scalar, 1x1) shape');
assert(all(b(:) == 1), 'ones values');

% Both dims as 1x1 arrays.
c = zeros([3], [2]);
assert(isequal(size(c), [3 2]), 'zeros(1x1, 1x1) shape');

% Single size-vector argument still builds the full shape.
d = zeros([2 6]);
assert(isequal(size(d), [2 6]), 'zeros(size-vector) shape');

% nan / Inf families share the same path.
e = nan([1], 3);
assert(isequal(size(e), [1 3]) && all(isnan(e(:))), 'nan(1x1, scalar)');

% A multi-element vector with extra args is an error (Size inputs must be scalar).
threw = false;
try
    zeros([2 3], 4);
catch
    threw = true;
end
assert(threw, 'zeros([2 3], 4) must error');

disp('SUCCESS');
