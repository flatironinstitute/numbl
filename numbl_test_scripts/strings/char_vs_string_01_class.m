% Test 1: Basic type identity - char vs string
% In MATLAB:
%   single-quoted literals 'hello' are char arrays
%   double-quoted literals "hello" are string scalars

% class()
assert(strcmp(class('hello'), 'char'));
assert(strcmp(class("hello"), 'string'));

% ischar()
assert(ischar('hello'));
assert(~ischar("hello"));

% isstring()
assert(~isstring('hello'));
assert(isstring("hello"));

disp('SUCCESS')
