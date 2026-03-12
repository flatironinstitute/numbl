% Test cellfun with 'UniformOutput', false

% Test 1: Returns a cell array when UniformOutput is false
C = {1, 2, 3};
result = cellfun(@(x) x > 1, C, 'UniformOutput', false);
assert(iscell(result), 'result should be a cell array');
assert(isequal(result{1}, false), 'first element should be false');
assert(isequal(result{2}, true), 'second element should be true');
assert(isequal(result{3}, true), 'third element should be true');

% Test 2: UniformOutput true (default) returns a regular array
result2 = cellfun(@(x) x > 1, C, 'UniformOutput', true);
assert(~iscell(result2), 'result should not be a cell array');
assert(isequal(result2, [false, true, true]), 'result2 should be logical array');

% Test 3: UniformOutput false with non-scalar results
result3 = cellfun(@(x) [x, x], C, 'UniformOutput', false);
assert(iscell(result3), 'result3 should be a cell array');
assert(isequal(result3{1}, [1, 1]), 'first element should be [1, 1]');
assert(isequal(result3{2}, [2, 2]), 'second element should be [2, 2]');

disp('SUCCESS')
