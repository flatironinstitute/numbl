% Test overloaded subsasgn on class instances.
% This mirrors the pattern used by chebfunpref: dot-access on a class
% instance should route through the user-defined subsasgn method.

% Test 1: Single-level dot assignment routes through subsasgn
obj = PrefStore_();
assert(obj.alpha == 10, 'Default alpha should be 10');
obj.alpha = 42;
assert(obj.alpha == 42, 'Single-level subsasgn should update alpha');

% Test 2: Multi-level dot assignment routes through subsasgn
% (This is the pattern that caused the chebfunpref bug)
obj2 = PrefStore_();
assert(obj2.opts.x == 1, 'Default opts.x should be 1');
assert(obj2.opts.y == 2, 'Default opts.y should be 2');
assert(obj2.opts.z == 3, 'Default opts.z should be 3');
obj2.opts.x = 99;
assert(obj2.opts.x == 99, 'Multi-level subsasgn should update opts.x');
% Crucially: the other fields must survive the assignment
assert(obj2.opts.y == 2, 'Multi-level subsasgn must preserve opts.y');
assert(obj2.opts.z == 3, 'Multi-level subsasgn must preserve opts.z');

% Test 3: builtin('subsasgn', ...) on a plain struct preserves fields
s.a = struct();
s.a.p = 10;
s.a.q = 20;
s.a.r = 30;
ind = struct('type', '.', 'subs', 'p');
s2 = builtin('subsasgn', s, struct('type', '.', 'subs', 'a'), struct('p', 99, 'q', 20, 'r', 30));
% Alternative: direct nested field assignment on plain struct (sanity check)
s.a.p = 99;
assert(s.a.p == 99, 'Plain struct nested assignment should work');
assert(s.a.q == 20, 'Plain struct other fields should survive');
assert(s.a.r == 30, 'Plain struct other fields should survive');

disp('SUCCESS');
