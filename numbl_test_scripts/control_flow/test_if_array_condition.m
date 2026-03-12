% Test that "if array" evaluates as true when ALL elements are nonzero
% In MATLAB, "if X" where X is an array is equivalent to "if all(X(:))"

% Logical vector - all true
result = false;
if [true, true]
    result = true;
end
assert(result, 'if [true,true] should enter the branch');

% Numeric array - all nonzero
result = false;
if [1, 2, 3]
    result = true;
end
assert(result, 'if [1,2,3] should enter the branch');

% Comparison producing logical array - all true
result = false;
if [1, 1] == 1
    result = true;
end
assert(result, 'if [1,1]==1 should enter the branch');

% size() == 1 pattern (common in real code)
A = 42;
result = false;
if size(A) == 1
    result = true;
end
assert(result, 'if size(scalar)==1 should enter the branch');

% One false element - should NOT enter
result = false;
if [true, false]
    result = true;
end
assert(~result, 'if [true,false] should NOT enter the branch');

% One zero element - should NOT enter
result = false;
if [1, 0, 3]
    result = true;
end
assert(~result, 'if [1,0,3] should NOT enter the branch');

% Also works for while
count = 0;
x = [1, 1];
while x
    count = count + 1;
    x = [0, 0];
end
assert(count == 1, 'while [1,1] should enter once');

disp('SUCCESS');
