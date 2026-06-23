% Old-style class with delegating subsref/subsasgn (the GeoPDEs pattern):
% overloads that just call builtin(...) to expose otherwise-private fields.
% Verifies external field read/write goes through the overload, while
% in-method dot-access uses default access (no recursion).

w = widget('a', 42);

% In-method default field access
assert(getval(w) == 42);

% External field read/write dispatch through the delegating subsref/subsasgn
assert(strcmp(w.name, 'a'));
assert(w.val == 42);
w.val = 99;
assert(w.val == 99);
assert(getval(w) == 99);

% Method that mutates a field, value semantics preserved
w2 = bump(w, 1);
assert(w2.val == 100);
assert(w.val == 99);

disp('SUCCESS')
