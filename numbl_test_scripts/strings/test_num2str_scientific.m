% Test num2str default formatting uses scientific notation for small/large numbers
% MATLAB's num2str uses short-g style: ~5 sig digits, scientific when exp < -4 or >= 5

% Small numbers that should use scientific notation
assert(strcmp(num2str(1e-5), '1e-05'));
assert(strcmp(num2str(1.234e-5), '1.234e-05'));

% Small numbers that should NOT use scientific notation
assert(strcmp(num2str(1e-4), '0.0001'));
assert(strcmp(num2str(0.001), '0.001'));
assert(strcmp(num2str(0.001234), '0.001234'));

% Regular numbers
assert(strcmp(num2str(0), '0'));
assert(strcmp(num2str(1), '1'));
assert(strcmp(num2str(42), '42'));
assert(strcmp(num2str(3.14), '3.14'));

% Special values
assert(strcmp(num2str(Inf), 'Inf'));
assert(strcmp(num2str(-Inf), '-Inf'));
assert(strcmp(num2str(NaN), 'NaN'));

disp('SUCCESS');
