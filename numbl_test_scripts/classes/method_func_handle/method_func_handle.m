% Test that @methodName inside a class method file uses runtime dispatch,
% and that anonymous functions capturing local helpers work cross-file.
%
% Test 1 (mimics chebfun sin/compose):
%   - Class method myop.m calls doWork(obj.value, @myop)
%   - doWork calls op(val) where val is a number
%   - @myop should dispatch to workspace myop.m (for numbers), not to
%     the class method (which only handles class instances)
%
% Test 2 (mimics chebfun compose/composeNested1):
%   - Class method processVals.m creates @(x) doubleIt(x) where
%     doubleIt is a local function in processVals.m
%   - The anonymous function is passed to applyOp (a different file)
%   - When applyOp calls the anonymous function, doubleIt must still
%     be resolvable as a local function of processVals.m

obj = FuncHandleClass_(5);

% Test 1: @myop runtime dispatch
result = myop(obj);
assert(result.value == 10);

% Test 2: anonymous function with local helper across files
result2 = processVals(obj);
assert(result2.value == 10);

disp('SUCCESS');
