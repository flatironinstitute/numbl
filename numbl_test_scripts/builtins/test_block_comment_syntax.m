% Test that %{ with trailing content is treated as a line comment, not a block comment
% In MATLAB, %{ only starts a block comment when it's alone on the line.
% If there's content after %{, it's a regular line comment.

%{ this should be a line comment, not a block comment }
x = 42;
assert(x == 42, 'assignment after %{ line comment failed');

% Also test that real block comments still work
%{
This is a real block comment
spanning multiple lines
%}
y = 10;
assert(y == 10, 'assignment after real block comment failed');

% Test %{ with just a closing brace on same line (like chebfun's subsref.m)
%{ }
z = 99;
assert(z == 99, 'assignment after %{ } line comment failed');

% Test %{ with other trailing text
%{ not a block comment
w = 7;
assert(w == 7, 'assignment after %{ with trailing text failed');

disp('SUCCESS');
