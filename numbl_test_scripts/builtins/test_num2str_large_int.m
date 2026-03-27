% Test num2str with large integers
% MATLAB: integer values should not use scientific notation

assert(strcmp(num2str(123456), '123456'), 'num2str(123456)');
assert(strcmp(num2str(1234567890), '1234567890'), 'num2str(1234567890)');
assert(strcmp(num2str(-1234567890), '-1234567890'), 'num2str(-1234567890)');
assert(strcmp(num2str(0), '0'), 'num2str(0)');
assert(strcmp(num2str(100000), '100000'), 'num2str(100000)');

% Non-integers should still use appropriate formatting
assert(strcmp(num2str(3.14159, 3), '3.14'), 'num2str with precision');

disp('SUCCESS');
