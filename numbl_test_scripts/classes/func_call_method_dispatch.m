% Test function-call syntax for class methods vs local functions
% In MATLAB, LOCAL functions take precedence over class methods
% when using function-call syntax. Dot syntax always goes to class method.

% Local function with same name (takes precedence over class method)
function r = double_it(x)
  r = x + x;
end

% Test 1: dot syntax -> class method
obj = Doubler_(3);
assert(obj.double_it(5) == 15);  % 5 * 3 = 15

% Test 2: function-call syntax -> local function (NOT class method)
result = double_it(7);
assert(result == 14);  % 7 + 7 = 14 (local function)

% Test 3: dot syntax still works after local function call
assert(obj.double_it(10) == 30);  % 10 * 3 = 30

disp('SUCCESS')
