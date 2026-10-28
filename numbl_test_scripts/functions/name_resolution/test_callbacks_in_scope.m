% Functions called by name or handle resolve from the caller's scope:
% a class method sees its classdef file's local functions, a private
% function sees its own locals and its private siblings, and a
% string-named call (feval) does not see nested functions.

o = NrClass();
assert(o.viaFeval() == 40);
assert(o.viaFevalString() == 60);
assert(o.viaHandle() == 50);
assert(isequal(o.viaCellfunHandle(), [10 20 30]));
assert(isequal(o.viaArrayfunHandle(), [10 20]));
assert(isequal(o.viaCellfunLegacyName(), [true false]));

r = nr_priv(1);
assert(r.fevalLocal == 101);
assert(isequal(r.arrayfunLocal, [101 101]));
assert(r.fevalSibling == 1001);
assert(r.cellfunSibling == 1001);

c = nr_caller();
assert(c.callNested == 3);
assert(c.handleNested == 7);
assert(c.fevalNestedFails);
assert(c.fevalLocal == -7);

% The script sees the private functions of its folder.
assert(feval('nr_priv2', 2) == 1002);

disp('SUCCESS')
