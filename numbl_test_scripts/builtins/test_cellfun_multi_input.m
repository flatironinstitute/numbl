% Test cellfun with multiple input cell arrays
% cellfun(func, C1, C2, ...) applies func to corresponding elements

% Test 1: two input cell arrays with plus
A = {1, 2, 3};
B = {10, 20, 30};
C = cellfun(@plus, A, B);
assert(isequal(C, [11, 22, 33]));

% Test 2: two input cell arrays with UniformOutput false
D = cellfun(@plus, A, B, 'UniformOutput', false);
assert(D{1} == 11);
assert(D{2} == 22);
assert(D{3} == 33);

% Test 3: three input cell arrays
X = {1, 2};
Y = {3, 4};
Z = {5, 6};
R = cellfun(@(a,b,c) a+b+c, X, Y, Z);
assert(isequal(R, [9, 12]));

fprintf('SUCCESS\n');
