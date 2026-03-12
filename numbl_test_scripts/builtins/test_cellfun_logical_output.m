% Test that cellfun returns logical when the applied function returns logical

% cellfun with @isempty should return logical
c = {[], 'a', [], 'b'};
result = cellfun(@isempty, c);
assert(isa(result, 'logical'), sprintf('Expected logical from cellfun(@isempty), got %s', class(result)));
assert(isequal(result, [true false true false]));

% Logical indexing deletion using cellfun result
c2 = {10, 20, 30, 40};
mask = cellfun(@isempty, {[], 'x', [], 'y'});
c2(mask) = [];
assert(numel(c2) == 2, sprintf('Expected 2 elements after deletion, got %d', numel(c2)));
assert(c2{1} == 20);
assert(c2{2} == 40);

% cellfun with @islogical should return logical
c3 = {true, 1, false, 'a'};
result2 = cellfun(@islogical, c3);
assert(isa(result2, 'logical'));
assert(isequal(result2, [true false true false]));

% cellfun with @isnumeric should return logical
c4 = {1, 'a', 2, 'b'};
result3 = cellfun(@isnumeric, c4);
assert(isa(result3, 'logical'));
assert(isequal(result3, [true false true false]));

% cellfun with numeric-returning function should return double
result4 = cellfun(@numel, {[1 2], [3 4 5], [6]});
assert(isa(result4, 'double'));
assert(isequal(result4, [2 3 1]));

disp('SUCCESS');
