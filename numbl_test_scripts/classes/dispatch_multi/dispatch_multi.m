% Test: multiple classes with same-named method + file function
% MATLAB rule: class method > file function. When multiple classes have
% the same method name, dispatch goes to whichever class the first arg
% is an instance of.
%
% AdderD_ has apply_(obj, x) -> obj.Val + x
% MultiplierD_ has apply_(obj, x) -> obj.Val * x
% File function apply_(x) -> -x
%
% apply_(adder, 5) -> AdderD_ method -> Val + 5
% apply_(multiplier, 5) -> MultiplierD_ method -> Val * 5
% apply_(3) -> file function -> -3

a = AdderD_(10);
m = MultiplierD_(10);

% Test 1: dot syntax (always works)
assert(a.apply_(5) == 15);   % 10 + 5
assert(m.apply_(5) == 50);   % 10 * 5

% Test 2: function-call syntax dispatches to correct class
assert(apply_(a, 5) == 15);  % AdderD_ method: 10 + 5
assert(apply_(m, 5) == 50);  % MultiplierD_ method: 10 * 5

% Test 3: function-call syntax with number -> file function
assert(apply_(3) == -3);

% Test 4: unknown types dispatch to correct class at runtime
X = 0;
X = AdderD_(20);
assert(apply_(X, 1) == 21);  % AdderD_ method: 20 + 1

Y = 0;
Y = MultiplierD_(3);
assert(apply_(Y, 7) == 21);  % MultiplierD_ method: 3 * 7

% Test 5: unknown type, runtime is number -> file function
Z = MultiplierD_(1);
Z = 5;
assert(apply_(Z) == -5);  % file function: -5

disp('SUCCESS')
