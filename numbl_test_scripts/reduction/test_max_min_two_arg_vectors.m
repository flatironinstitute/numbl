% Test that max(A, B) and min(A, B) with two vector arguments
% perform element-wise comparison and return a vector.

% Test 1: max with two row vectors
a = [1 5 3 7];
b = [4 2 6 1];
result = max(a, b);
assert(isequal(result, [4 5 6 7]), 'max of two row vectors');

% Test 2: min with two row vectors
result2 = min(a, b);
assert(isequal(result2, [1 2 3 1]), 'min of two row vectors');

% Test 3: max with two column vectors
c = [1; 5; 3];
d = [4; 2; 6];
result3 = max(c, d);
assert(isequal(result3, [4; 5; 6]), 'max of two column vectors');

% Test 4: min with two column vectors
result4 = min(c, d);
assert(isequal(result4, [1; 2; 3]), 'min of two column vectors');

% Test 5: max with scalar and vector (scalar expansion)
result5 = max([1 5 3], 4);
assert(isequal(result5, [4 5 4]), 'max of vector and scalar');

% Test 6: min with scalar and vector (scalar expansion)
result6 = min([1 5 3], 4);
assert(isequal(result6, [1 4 3]), 'min of vector and scalar');

% Test 7: max with two matrices
A = [1 2; 3 4];
B = [4 1; 2 5];
result7 = max(A, B);
assert(isequal(result7, [4 2; 3 5]), 'max of two matrices');

% Test 8: The chebfun pattern - max(max(abs(x(:)), abs(y(:))))
x = [1 -5 3; -7 2 4];
y = [-2 3 -6; 1 -8 5];
inner = max(abs(x(:)), abs(y(:)));
outer = max(inner);
assert(outer == 8, 'nested max with colon indexing pattern');

disp('SUCCESS');
