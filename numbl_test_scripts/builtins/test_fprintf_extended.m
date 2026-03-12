% Test fprintf with various format strings (specialBuiltins.ts coverage)

% Basic string
fprintf('hello\n');

% Format with integer
fprintf('x = %d\n', 42);

% Format with float
fprintf('pi = %.4f\n', 3.14159);

% Format with string
fprintf('name = %s\n', 'world');

% Multiple format specifiers
fprintf('%d + %d = %d\n', 1, 2, 3);

% Tensor argument expansion
fprintf('%d ', [1 2 3 4]);
fprintf('\n');

% No format specifiers, just a string
fprintf('plain text\n');

disp('SUCCESS');
