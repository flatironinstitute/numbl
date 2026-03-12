% Test that sprintf flattens array arguments (MATLAB behavior)

% Basic array flattening: vector arg expands to multiple scalars
s1 = sprintf('%g %g', [3.14, 2.71]);
assert(strcmp(s1, '3.14 2.71'), sprintf('Test 1 failed: got "%s"', s1));

% Format string repeated for extra elements
s2 = sprintf('%d,', [1, 2, 3]);
assert(strcmp(s2, '1,2,3,'), sprintf('Test 2 failed: got "%s"', s2));

% Column-major order for matrices
s3 = sprintf('%d ', [1 2; 3 4]);
assert(strcmp(s3, '1 3 2 4 '), sprintf('Test 3 failed: got "%s"', s3));

% Mixed scalar and array args (endvals scenario from chebfun)
endvals = [1.5, 2.5];
s4 = sprintf('[%8.2g,%8.2g]   %6i  %8.2g %8.2g', 0, 1, 10, endvals);
expected4 = sprintf('[%8.2g,%8.2g]   %6i  %8.2g %8.2g', 0, 1, 10, 1.5, 2.5);
assert(strcmp(s4, expected4), sprintf('Test 4 failed: got "%s"', s4));

% Single element tensor (should work as before)
s5 = sprintf('%d', [42]);
assert(strcmp(s5, '42'), sprintf('Test 5 failed: got "%s"', s5));

% Empty format args
s6 = sprintf('hello');
assert(strcmp(s6, 'hello'), sprintf('Test 6 failed: got "%s"', s6));

disp('SUCCESS');
