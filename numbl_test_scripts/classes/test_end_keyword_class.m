% Test that the `end` keyword in indexing expressions calls the class's
% overloaded end() method and that deferred ranges (e.g. 2:end) are
% properly resolved before being passed to subsref.

obj = EndTracker_([10 20 30 40]);

% Test 1: obj(end) — bare end as single index
result1 = obj(end);
assert(result1 == 40, 'obj(end) should return last element');

% Test 2: obj(2:end) — range with end
result2 = obj(2:end);
assert(isequal(result2, [20 30 40]), 'obj(2:end) should return elements 2 through 4');

% Test 3: obj(1:end-1) — compound expression with end
result3 = obj(1:end-1);
assert(isequal(result3, [10 20 30]), 'obj(1:end-1) should return elements 1 through 3');

% Test 4: 2D data with end in second dimension
obj2 = EndTracker_([1 2 3; 4 5 6]);

result4 = obj2(:, 2:end);
assert(isequal(result4, [2 3; 5 6]), 'obj(:,2:end) should return columns 2 through 3');

result5 = obj2(:, end);
assert(isequal(result5, [3; 6]), 'obj(:,end) should return last column');

disp('SUCCESS');
