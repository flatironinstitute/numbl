%% cellfun with 'isempty' string function
c = {1, [], 'hello', {}};
result = cellfun(@isempty, c);
assert(isequal(result, [false, true, false, true]));

%% cellfun with 'islogical' string function
c2 = {true, 1, false, 'hi'};
result2 = cellfun(@islogical, c2);
assert(isequal(result2, [true, false, true, false]));

%% cellfun with 'isreal' string function
c3 = {1, 2+3i, [1 2], 'abc'};
result3 = cellfun(@isreal, c3);
assert(isequal(result3, [true, false, true, true]));

%% cellfun with 'length' string function
c4 = {[1 2 3], 'ab', {1, 2, 3, 4}, 5};
result4 = cellfun(@length, c4);
assert(isequal(result4, [3, 2, 4, 1]));

%% cellfun with 'ndims' string function
c5 = {1, [1 2; 3 4], ones(2,3,4)};
result5 = cellfun(@ndims, c5);
assert(isequal(result5, [2, 2, 3]));

%% cellfun('isclass', ..., className)
c6 = {1, 'hello', true, [1 2]};
result6 = cellfun('isclass', c6, 'double');
assert(isequal(result6, [true, false, false, true]));

result7 = cellfun('isclass', c6, 'char');
assert(isequal(result7, [false, true, false, false]));

result8 = cellfun('isclass', c6, 'logical');
assert(isequal(result8, [false, false, true, false]));

disp('SUCCESS')
