% Test: class method shadowing a builtin function
% MATLAB rule: class methods take precedence over builtins
%
% SizedD_ class has custom "size(obj)" that returns -42
% "size" is also a builtin that returns array dimensions
%
% size(obj) should call CLASS METHOD -> -42
% size([1 2 3]) should call BUILTIN -> [1 3]
% obj.size() should call class method -> -42

s = SizedD_(100);

% Test 1: dot syntax -> class method
assert(s.size() == -42);

% Test 2: function-call syntax with class instance -> class method
assert(size(s) == -42);

% Test 3: function-call syntax with array -> builtin
v = [1 2 3 4 5];
sz = size(v);
assert(sz(1) == 1);
assert(sz(2) == 5);

% Test 4: custom length method
assert(s.length() == -99);
assert(length(s) == -99);

% Test 5: builtin length with array
assert(length(v) == 5);

% Test 6: unknown type, runtime is class instance
X = 0;
X = SizedD_(50);
assert(size(X) == -42);

% Test 7: unknown type, runtime is array
Y = SizedD_(1);
Y = [10 20 30];
sz2 = size(Y);
assert(sz2(1) == 1);
assert(sz2(2) == 3);

disp('SUCCESS')
