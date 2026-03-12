% Test: local function takes precedence over class method in function-call syntax
% MATLAB rule: local functions > class methods > file functions > builtins
%
% ScalerD_ class has method "transform(obj, x)" that returns x * obj.Factor
% Local function "transform(obj, x)" always returns -1
%
% transform(s, 10) should call LOCAL function -> -1
% s.transform(10) should call class method -> 50

s = ScalerD_(5);

% Test 1: dot syntax -> class method (always)
assert(s.transform(10) == 50);  % class method: 10 * 5

% Test 2: function-call syntax -> local function (takes precedence)
r = transform(s, 10);
assert(r == -1);  % local function always returns -1

% Test 3: function-call syntax with number -> also local function
r2 = transform(42, 7);
assert(r2 == -1);

disp('SUCCESS')

function r = transform(~, ~)
  r = -1;
end
