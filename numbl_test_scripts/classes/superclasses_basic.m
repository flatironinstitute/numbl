% Test the superclasses() builtin: returns a column cell array of the
% names of a class's ancestors (immediate parent first), or a 0x1 empty
% cell for built-in types / classes with no superclasses.

% --- Object input, 3-level chain: child -> parent -> grandparent ---
child = SuperChild_(1, 2, 3);
s = superclasses(child);
assert(iscell(s));
assert(isequal(size(s), [2, 1]));
assert(strcmp(s{1}, 'SuperParent_'));
assert(strcmp(s{2}, 'SuperGrandparent_'));

% --- Class-name (char) input gives the same result ---
s2 = superclasses('SuperChild_');
assert(isequal(s, s2));

% --- Middle of the chain: one superclass ---
p = SuperParent_(1, 2);
sp = superclasses(p);
assert(isequal(size(sp), [1, 1]));
assert(strcmp(sp{1}, 'SuperGrandparent_'));

% --- Top of the chain: no superclasses -> 0x1 empty cell ---
g = SuperGrandparent_(1);
sg = superclasses(g);
assert(iscell(sg));
assert(isempty(sg));
assert(isequal(size(sg), [0, 1]));

% --- Built-in numeric value -> empty cell ---
sn = superclasses(rand(3, 4));
assert(iscell(sn));
assert(isempty(sn));

% --- Built-in class name -> empty cell ---
sd = superclasses('double');
assert(iscell(sd));
assert(isempty(sd));

disp('SUCCESS')
