% Test that strncmp/strncmpi return false for non-string inputs
% instead of throwing an error.

f = @(x) x + 1;

% strncmp with function handle should return false
assert(strncmp(f, '--', 2) == 0, 'strncmp(func, str, n) should return false');
assert(strncmp('--hello', f, 2) == 0, 'strncmp(str, func, n) should return false');

% strncmpi with function handle should return false
assert(strncmpi(f, '--', 2) == 0, 'strncmpi(func, str, n) should return false');
assert(strncmpi('--hello', f, 2) == 0, 'strncmpi(str, func, n) should return false');

% strncmp with numeric input should return false
assert(strncmp(42, 'hello', 3) == 0, 'strncmp(num, str, n) should return false');
assert(strncmp('hello', 42, 3) == 0, 'strncmp(str, num, n) should return false');

% strncmpi with numeric input should return false
assert(strncmpi(42, 'hello', 3) == 0, 'strncmpi(num, str, n) should return false');
assert(strncmpi('hello', 42, 3) == 0, 'strncmpi(str, num, n) should return false');

disp('SUCCESS');
