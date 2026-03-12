% Test isequal with mixed types (logical/double, scalar/tensor)
% MATLAB's isequal compares values, not internal type representations

% logical vs double
assert(isequal(1, true), 'isequal(1, true)');
assert(isequal(true, 1), 'isequal(true, 1)');
assert(isequal(0, false), 'isequal(0, false)');
assert(isequal(false, 0), 'isequal(false, 0)');
assert(~isequal(1, false), 'isequal(1, false) should be false');

% scalar number vs 1-element range tensor
assert(isequal(5:5, 5), 'isequal(5:5, 5)');
assert(isequal(5, 5:5), 'isequal(5, 5:5)');

% 1-element tensor from range vs array literal
assert(isequal(5:5, [5]), 'isequal(5:5, [5])');

% logical vector vs double vector
assert(isequal([1 0 1], [true false true]), 'isequal double vec vs logical vec');

% logical scalar vs 1-element tensor
assert(isequal(true, [1]), 'isequal(true, [1])');
assert(isequal([0], false), 'isequal([0], false)');

disp('SUCCESS');
