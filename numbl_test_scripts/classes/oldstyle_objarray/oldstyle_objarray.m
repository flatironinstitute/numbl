% Old-style class object arrays: a constructor that grows an array of its own
% class in an (initially undefined) struct field, with chained assignment into
% the elements — the pattern @msh_cartesian uses to build msh.boundary.

m = cell2d(3);

assert(strcmp(class(m), 'cell2d'));
assert(m.ndim == 3);

% boundary grew into a 2-element object array
b = m.boundary;
assert(strcmp(class(b), 'cell2d'));
assert(numel(b) == 2);

% Element indexing + field access through subsref
assert(strcmp(class(m.boundary(1)), 'cell2d'));
assert(m.boundary(1).ndim == 2);
assert(m.boundary(1).val == 901);   % chained assign set 900+iside
assert(m.boundary(2).val == 902);

% Method dispatch on an indexed element
assert(getval(m.boundary(2)) == 902);

disp('SUCCESS')
