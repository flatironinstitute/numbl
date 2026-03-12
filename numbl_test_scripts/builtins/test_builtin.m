% Test the builtin() function

% Test 1: builtin with string name
x = builtin('zeros', 1, 3);
assert(isequal(x, [0 0 0]), 'builtin zeros failed');

% Test 2: builtin with ones and multiple args
x = builtin('ones', 2, 3);
assert(isequal(size(x), [2 3]), 'builtin ones size failed');
assert(all(all(x == 1)), 'builtin ones values failed');

% Test 3: builtin with eye
x = builtin('eye', 3);
assert(isequal(size(x), [3 3]), 'builtin eye size failed');
assert(x(1,1) == 1 && x(2,2) == 1 && x(1,2) == 0, 'builtin eye values failed');

% Test 4: builtin result can be used in expressions
x = builtin('ones', 1, 4) * 3;
assert(isequal(x, [3 3 3 3]), 'builtin in expression failed');

% Test 5: builtin with numel
x = builtin('numel', [1 2; 3 4; 5 6]);
assert(x == 6, 'builtin numel failed');

% Test 6: builtin with length
x = builtin('length', [1 2 3 4]);
assert(x == 4, 'builtin length failed');

% Test 7: builtin calls the real builtin even when a local function shadows it
x = zeros(1, 3);
assert(isequal(x, [99 99 99]), 'local zeros should return 99s');
x = builtin('zeros', 1, 3);
assert(isequal(x, [0 0 0]), 'builtin zeros should bypass local override');

fprintf('SUCCESS\n');

function y = zeros(varargin)
    % Local function that shadows the builtin zeros
    y = ones(varargin{:}) * 99;
end
