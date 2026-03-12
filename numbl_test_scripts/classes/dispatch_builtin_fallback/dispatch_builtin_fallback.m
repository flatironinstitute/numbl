% Test: builtin fallback when class does NOT define a method
% When numel(obj) is called and the class has no numel method,
% it should fall back to the built-in numel function.

obj = NoNumel_(42);

% Test 1: numel on class instance (known type) — should use builtin
assert(numel(obj) == 1);

% Test 2: numel on class instance (unknown type at compile time)
x = 0;
x = NoNumel_(10);
assert(numel(x) == 1);

% Test 3: size on class instance — should use builtin
sz = size(obj);
assert(sz(1) == 1);
assert(sz(2) == 1);

% Test 4: length on class instance — should use builtin
assert(length(obj) == 1);

disp('SUCCESS')
