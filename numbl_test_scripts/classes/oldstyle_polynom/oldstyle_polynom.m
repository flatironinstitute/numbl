% Old-style (pre-classdef) class: construction, introspection, method
% dispatch (function-call form), value semantics, leftmost-object dispatch,
% and operator overloading. Runs unchanged in numbl and MATLAB.

p = polynom([1 2 3]);

% Introspection
assert(strcmp(class(p), 'polynom'));
assert(isa(p, 'polynom'));
assert(~isa(p, 'double'));
assert(isobject(p));
assert(~isstruct(p));
assert(isscalar(p));
assert(numel(p) == 1);
assert(isequal(fieldnames(p), {'c'}));

% Function-call method dispatch
assert(isequal(coeffs(p), [1 2 3]));
assert(isequal(double(p), [1 2 3]));

% Value semantics: scale returns a modified copy; original unchanged
q = scale(p, 10);
assert(isequal(coeffs(q), [10 20 30]));
assert(isequal(coeffs(p), [1 2 3]));

% Copy constructor idiom: polynom(p) returns p unchanged
r = polynom(p);
assert(isequal(coeffs(r), [1 2 3]));

% Operator overloading (dispatch on the object operand)
s = p + polynom([1 0 0]);
assert(isequal(coeffs(s), [2 2 3]));
t = 5 + p;                  % 5 promoted via polynom() in plus.m
assert(isequal(coeffs(t), [1 2 8]));

% Leftmost-object dispatch: combine exists in @polynom and @dualnum.
d = dualnum(7);
assert(strcmp(combine(p, d), 'polynom'));   % leftmost is polynom
assert(strcmp(combine(d, p), 'dualnum'));   % leftmost is dualnum
assert(strcmp(combine(3, d), 'dualnum'));   % first object arg wins

% Build an object array from an undefined variable (no subsasgn overload).
clear arr
arr(1) = polynom([1 1]);
arr(2) = polynom([2 2]);
arr(3) = polynom([3 3]);
assert(numel(arr) == 3);
assert(isa(arr, 'polynom'));
assert(isequal(coeffs(arr(3)), [3 3]));

disp('SUCCESS')
