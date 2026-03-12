% Test: class method subfunctions have independent variable scope
% A local subfunction in a class method file should NOT share variables
% with the primary method. MATLAB subfunctions have their own workspace.

obj = ScopeTest_(10);
result = compute(obj);
% helper(10) returns 10+2=12, then result = 12+5=17
assert(result == 17, sprintf('Expected 17 but got %d', result));

disp('SUCCESS')
