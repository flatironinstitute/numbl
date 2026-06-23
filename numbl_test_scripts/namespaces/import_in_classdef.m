% A wildcard import inside a classdef method body must resolve a bare
% package-function call.

obj = ImportMethodWild_;
assert(obj.compute(5) == 12, 'classdef-method wildcard import');

disp('SUCCESS')
