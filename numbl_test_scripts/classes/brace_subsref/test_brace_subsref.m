% Test curly-brace {} indexing on class instances via subsref
% In MATLAB, obj{a,b,c} dispatches to subsref with type '{}'

obj = BraceTest(42);

% Test 1: single brace index
r1 = obj{10};
assert(r1 == 10);

% Test 2: multiple brace indices (sum them)
r2 = obj{1, 2, 3};
assert(r2 == 6);

% Test 3: data property still accessible
assert(obj.data == 42);

fprintf('SUCCESS\n');
