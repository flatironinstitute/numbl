% Test: file-local subfunctions in classdef method files should use
% built-in indexing (not overloaded subsref) for instances of the same class.

o = SubsrefLocalFn_(7);

% External access: o(1) goes through overloaded subsref (doubles value)
assert(o(1) == 14);      % 7 * 2 = 14

% Method delegates to a file-local subfunction:
% Inside the local subfunction, obj(1) should use built-in indexing
assert(o.getViaHelper() == 7);  % NOT 14

disp('SUCCESS');
