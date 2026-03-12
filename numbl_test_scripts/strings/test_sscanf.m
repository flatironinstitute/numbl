% Test sscanf

%% Basic integer
val = sscanf('42', '%d');
assert(val == 42);

%% Basic float
val = sscanf('3.14', '%f');
assert(abs(val - 3.14) < 1e-10);

%% Multiple values
vals = sscanf('1 2 3', '%d');
assert(isequal(vals, [1; 2; 3]));

%% Mixed format
vals = sscanf('hello 42 3.14', '%s %d %f');
% sscanf with %s reads one word
assert(vals(1) == 'h');  % sscanf returns chars as ASCII codes in the column

%% Format with literals
vals = sscanf('x=10,y=20', 'x=%d,y=%d');
assert(isequal(vals, [10; 20]));

%% Hexadecimal
val = sscanf('ff', '%x');
assert(val == 255);

%% Octal
val = sscanf('77', '%o');
assert(val == 63);

%% Scientific notation
val = sscanf('1.5e3', '%f');
assert(val == 1500);

%% Multiple floats
vals = sscanf('1.1 2.2 3.3', '%f');
assert(length(vals) == 3);
assert(abs(vals(1) - 1.1) < 1e-10);
assert(abs(vals(2) - 2.2) < 1e-10);
assert(abs(vals(3) - 3.3) < 1e-10);

%% Size argument - read specific count
vals = sscanf('1 2 3 4 5', '%d', 3);
assert(isequal(vals, [1; 2; 3]));

%% Size argument with Inf reads all
vals = sscanf('1 2 3', '%d', Inf);
assert(isequal(vals, [1; 2; 3]));

%% Negative numbers
vals = sscanf('-5 -10', '%d');
assert(isequal(vals, [-5; -10]));

%% Single character read
val = sscanf('A', '%c');
assert(val == 'A');

disp('SUCCESS');
