% An old-style (pre-classdef) class whose constructor calls superiorto must
% construct, dispatch its methods, and take precedence over double in
% operator overloading.

p = superiorpoly([1 2 3 4]);
assert(isa(p, 'superiorpoly'), 'isa superiorpoly');
assert(isequal(getvec(p), [1; 2; 3; 4]), 'method field access');

% sum of stored vector is 10; the overloaded + must run with the object on
% either side (superiorpoly is superior to double).
assert(p + 10 == 20, 'obj + double');
assert(100 + p == 110, 'double + obj');

disp('SUCCESS')
