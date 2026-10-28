% nargin() on builtin handles: fixed-arity text builtins and variable-input
% builtins (which report a negative count).
assert(nargin(@strrep) == 3);
assert(nargin(@zeros) == -1);
assert(nargin(@ones) == -1);
assert(nargin(@cat) == -1);
assert(nargin(@linspace) == 3);
assert(nargin(@upper) == 1);
assert(nargin(@isfield) == 2);
disp('SUCCESS')
