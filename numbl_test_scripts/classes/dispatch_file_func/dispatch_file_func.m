% Test: class method takes precedence over file function
% MATLAB rule: local functions > class methods > file functions
%
% ShapeD_ class has method "area_(obj)" that returns obj.W * obj.H
% File function "area_(x)" in area_.m returns x * x
%
% Since area_.m is a FILE function (not local), class method takes precedence
% when the first arg is a class instance.
%
% area_(s) should call CLASS METHOD -> 12
% area_(5) should call file function -> 25
% s.area_() should call class method -> 12

s = ShapeD_(3, 4);

% Test 1: dot syntax -> class method
assert(s.area_() == 12);

% Test 2: function-call syntax with class instance -> class method
% (class method takes precedence over file function)
assert(area_(s) == 12);

% Test 3: function-call syntax with number -> file function
assert(area_(5) == 25);  % 5 * 5

% Test 4: unknown type, runtime is class instance -> class method
X = 0;
X = ShapeD_(6, 7);
assert(area_(X) == 42);

% Test 5: unknown type, runtime is number -> file function
Y = ShapeD_(1, 1);
Y = 3;
assert(area_(Y) == 9);  % 3 * 3

disp('SUCCESS')
