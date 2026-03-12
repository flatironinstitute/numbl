% Test colon operator with char values
% 'a':'e' should produce 'abcde' by treating chars as ASCII codes

% Basic char range
r = 'a':'e';
assert(isequal(r, 'abcde'));

% Uppercase range
r2 = 'A':'E';
assert(isequal(r2, 'ABCDE'));

% Single char range
r3 = 'x':'x';
assert(isequal(r3, 'x'));

% Char range with step
r4 = 'a':2:'g';
assert(isequal(r4, 'aceg'));

% Numeric result from char arithmetic
assert(double('a') == 97);
assert(double('A') == 65);

disp('SUCCESS');
