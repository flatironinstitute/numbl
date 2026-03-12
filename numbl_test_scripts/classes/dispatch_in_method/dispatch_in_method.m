% Test: function-call syntax inside class method bodies
%
% BoxD_ class has:
%   area_(obj) -> obj.W * obj.H
%   double_area_(obj) -> area_(obj) * 2   (calls area_ via function-call syntax)
%   area_plus_(obj, extra) -> obj.area_() + extra  (calls area_ via dot syntax)
%
% File function area_(x) -> x * x (in area_.m)
%
% Inside double_area_, area_(obj) should call the CLASS METHOD (not file function)
% because class methods take precedence over file functions.

b = BoxD_(3, 4);

% Test 1: dot syntax -> class method
assert(b.area_() == 12);

% Test 2: method that calls another method via function-call syntax
assert(b.double_area_() == 24);  % area_(obj) inside -> 12, * 2 = 24

% Test 3: method that calls another method via dot syntax
assert(b.area_plus_(8) == 20);  % area_ = 12, + 8 = 20

% Test 4: function-call syntax from outside with class instance
assert(area_(b) == 12);  % class method (precedence over file function)

% Test 5: function-call syntax with number -> file function
assert(area_(5) == 25);  % file function: 5 * 5

disp('SUCCESS')
