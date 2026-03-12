% Test max/min with empty arrays
% In MATLAB, max([]) and min([]) return []

% max of empty array
x1 = max([]);
assert(isempty(x1), 'max([]) should be empty');

% min of empty array
x2 = min([]);
assert(isempty(x2), 'min([]) should be empty');

% max with two outputs on empty
[v, i] = max([]);
assert(isempty(v), 'max([]) value should be empty');
assert(isempty(i), 'max([]) index should be empty');

% min with two outputs on empty
[v2, i2] = min([]);
assert(isempty(v2), 'min([]) value should be empty');
assert(isempty(i2), 'min([]) index should be empty');

disp('SUCCESS');
