% Test cell2mat with logical values in cells

% Test 1: cell of scalar logicals → logical row vector
C = {true, false, true};
result = cell2mat(C);
assert(isequal(result, [true, false, true]), 'cell2mat should handle logical scalars');

% Test 2: result from cellfun with UniformOutput false → cell of logicals
nums = {1, 2, 3};
flags = cellfun(@(x) x > 1, nums, 'UniformOutput', false);
result2 = cell2mat(flags);
assert(isequal(result2, [false, true, true]), 'cell2mat on cellfun logical results');

% Test 3: all() on result should work
assert(~all(result2), 'all() on mixed logical should be false');
assert(all(cell2mat({true, true})), 'all() on all-true should be true');

disp('SUCCESS')
