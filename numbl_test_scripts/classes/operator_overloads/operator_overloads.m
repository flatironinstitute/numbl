% Unary and logical operators on a classdef object call its overloads:
% ~a calls not, -a uminus, +a uplus, a' ctranspose, a.' transpose, and the
% functional forms reach the same methods.

a = OpVal([1 0 2]);

r = ~a;
assert(strcmp(r.tag, 'not'));
assert(isequal(r.v, [false true false]));
r = not(a);
assert(strcmp(r.tag, 'not'));

r = -a;
assert(strcmp(r.tag, 'uminus'));
assert(isequal(r.v, [-1 0 -2]));
r = uminus(a);
assert(strcmp(r.tag, 'uminus'));

r = +a;
assert(strcmp(r.tag, 'uplus'));
assert(isequal(r.v, [1 0 2]));

c = OpVal([1+2i 3]);
r = c';
assert(strcmp(r.tag, 'ctranspose'));
assert(isequal(r.v, [1-2i; 3]));
r = c.';
assert(strcmp(r.tag, 'transpose'));
assert(isequal(r.v, [1+2i; 3]));
r = ctranspose(c);
assert(strcmp(r.tag, 'ctranspose'));

% Binary logical operators, with the object on either side.
r = a & [1 1 0];
assert(strcmp(r.tag, 'and'));
assert(isequal(r.v, [true false false]));
r = [0 1 1] | a;
assert(strcmp(r.tag, 'or'));
assert(isequal(r.v, [true true true]));

% Operators on an object array dispatch to the class too.
arr = [OpVal(1) OpVal(2)];
r = arr + 1;
assert(strcmp(r.tag, 'plus'));
assert(isequal(r.v, [2 3]));

% A scalar object.
b = OpVal(0);
nb = ~b;
assert(nb.v == true);

disp('SUCCESS')
