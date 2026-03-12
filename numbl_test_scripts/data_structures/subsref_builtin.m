% Test: builtin subsref() function
% subsref(obj, S) applies subscript operations described by S to obj.

% Test 1: dot access on struct
s = struct('x', 10, 'y', 20);
S = struct('type', '.', 'subs', 'x');
assert(subsref(s, S) == 10, 'subsref dot access failed');

S2 = struct('type', '.', 'subs', 'y');
assert(subsref(s, S2) == 20, 'subsref dot access y failed');

% Test 2: paren indexing on array
x = [10, 20, 30];
S3 = struct;
S3.type = '()';
S3.subs = {2};
assert(subsref(x, S3) == 20, 'subsref paren indexing failed');

% Test 3: paren indexing with multiple subscripts (2D)
A = [1 2 3; 4 5 6; 7 8 9];
S4 = struct;
S4.type = '()';
S4.subs = {2, 3};
assert(subsref(A, S4) == 6, 'subsref 2D paren indexing failed');

% Test 4: brace indexing on cell array
c = {100, 'hello', 42};
S5 = struct;
S5.type = '{}';
S5.subs = {1};
assert(subsref(c, S5) == 100, 'subsref brace indexing failed');

disp('SUCCESS')
