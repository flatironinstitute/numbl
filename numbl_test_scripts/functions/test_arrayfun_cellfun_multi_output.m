% Test arrayfun and cellfun with multiple outputs
% In MATLAB, arrayfun/cellfun pass the correct nargout to the function handle

% arrayfun with two outputs
[a, b] = arrayfun(@(x) deal(x, x^2), [1 2 3]);
assert(isequal(a, [1 2 3]), 'arrayfun multi out a');
assert(isequal(b, [1 4 9]), 'arrayfun multi out b');

% cellfun with two outputs
[c, d] = cellfun(@(x) deal(x, x^2), {1 2 3});
assert(isequal(c, [1 2 3]), 'cellfun multi out c');
assert(isequal(d, [1 4 9]), 'cellfun multi out d');

% arrayfun with two inputs and two outputs
[q, r] = arrayfun(@(a, b) deal(floor(a/b), mod(a, b)), [10 20 30], [3 7 4]);
assert(isequal(q, [3 2 7]), 'arrayfun multi-input multi-output q');
assert(isequal(r, [1 6 2]), 'arrayfun multi-input multi-output r');

disp('SUCCESS');
