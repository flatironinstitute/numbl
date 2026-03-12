% Test num2str with numeric precision argument
% num2str(x, n) should format x with n significant digits

% Basic precision
assert(strcmp(num2str(3.14159, 3), '3.14'));
assert(strcmp(num2str(3.14159, 5), '3.1416'));
assert(strcmp(num2str(3.14159, 1), '3'));

% Small numbers
assert(strcmp(num2str(0.001234, 2), '0.0012'));

% Large numbers use scientific when needed
assert(strcmp(num2str(99999, 3), '1e+05'));

% Integer values
assert(strcmp(num2str(42, 5), '42'));

% Negative values
assert(strcmp(num2str(-3.14159, 3), '-3.14'));

% num2str with format string still works
assert(strcmp(strtrim(num2str(3.14, '%10.2f')), '3.14'));

disp('SUCCESS');
