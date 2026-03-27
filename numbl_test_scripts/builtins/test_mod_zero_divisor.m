% Test mod(a, 0) edge case
% In MATLAB, mod(a, 0) returns a (not NaN)

% Scalar cases
assert(mod(5, 0) == 5, 'mod(5,0) should be 5');
assert(mod(-3, 0) == -3, 'mod(-3,0) should be -3');
assert(mod(0, 0) == 0, 'mod(0,0) should be 0');
assert(mod(7.5, 0) == 7.5, 'mod(7.5,0) should be 7.5');

% Vector case
result = mod([5 -3 0 7.5], 0);
expected = [5 -3 0 7.5];
assert(isequal(result, expected), 'mod(vector, 0) should return the vector');

% Also verify normal mod still works
assert(mod(7, 3) == 1, 'mod(7,3) should be 1');
assert(mod(-7, 3) == 2, 'mod(-7,3) should be 2');
assert(mod(7, -3) == -2, 'mod(7,-3) should be -2');

disp('SUCCESS');
