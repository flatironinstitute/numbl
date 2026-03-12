% Test: class method dispatch should only apply when argument is a class instance.
% When isempty(x) is called inside a class method where x is NOT an instance
% of that class (e.g., x is a cell array), it should use the builtin isempty,
% not the class's overloaded isempty.

obj = MyContainer({1, 2, 3});

% Test 1: isempty on the class instance itself - should use @MyContainer/isempty
assert(~isempty(obj));

% Test 2: isempty on an empty class instance - should use @MyContainer/isempty
empty_obj = MyContainer();
assert(isempty(empty_obj));

% Test 3: doCheck with args - varargin is non-empty, builtin isempty should return false
result = doCheck(obj, 1, 2, 3);
assert(~result);

% Test 4: doCheck with no extra args - varargin is empty, builtin isempty should return true
result2 = doCheck(obj);
assert(result2);

disp('SUCCESS');
