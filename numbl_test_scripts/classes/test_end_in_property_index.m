% Test that the 'end' keyword resolves correctly when indexing into a
% class property via obj.property([1, end]).
% This pattern is used by chebfun: f(1).domain([1, end]).

obj = PropEndTest_([10 20 30 40 50]);

% Test 1: obj.data(end) should return last element
assert(obj.data(end) == 50, 'obj.data(end) failed');

% Test 2: obj.data([1, end]) should return first and last
result = obj.data([1, end]);
assert(isequal(result, [10, 50]), 'obj.data([1, end]) failed');

% Test 3: obj.data(2:end) should return elements 2 through 5
result2 = obj.data(2:end);
assert(isequal(result2, [20 30 40 50]), 'obj.data(2:end) failed');

disp('SUCCESS');
