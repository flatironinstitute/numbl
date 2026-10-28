% nargin('name') resolves the name in the caller's scope.

assert(nargin('nr_script_loc') == 2);
assert(nargin("nr_script_loc") == 2);
assert(nargin('nr_script_var') == -2);
assert(nargin(@nr_script_loc) == 2);
assert(nargin('nr_priv') == 1);
assert(nargin('nr_caller') == 0);
assert(nargin('nrpkg.nr_pkgfn') == 1);
assert(nargin('sin') == 1);

o = NrClass();
assert(o.narginLocal() == 1);

r = nr_priv(1);
assert(r.narginLocal == 2);

c = nr_caller();
assert(c.narginNested == 2);
assert(c.narginLocal == 1);

disp('SUCCESS')

function y = nr_script_loc(a, b)
    y = a + b;
end

function y = nr_script_var(a, varargin)
    y = a;
end
