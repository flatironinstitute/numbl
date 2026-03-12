% Test fprintf return value (number of bytes written)

% Simple string
n = fprintf('test');
assert(n == 4);

% With format specifiers
n = fprintf('%d', 42);
assert(n == 2);

% With newline
n = fprintf('hello\n');
assert(n == 6);  % 'hello' + newline = 6 chars

% With multiple args
n = fprintf('%d %d\n', 10, 20);
assert(n == 6);  % '10 20\n'

disp('SUCCESS');
