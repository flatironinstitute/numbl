% Test: InferiorClasses dispatch
%
% SuperiorObj_ declares InferiorClasses = {?BasicObj_}, meaning
% SuperiorObj_ is superior to BasicObj_ for method dispatch.
% When both classes have the same method (plus, minus, mtimes),
% MATLAB dispatches to SuperiorObj_'s method regardless of arg order.

a = BasicObj_(1);
s = SuperiorObj_(2);

%% plus: superior obj first
r = plus(s, a);
assert(strcmp(r, 'superior'), 'plus(superior, basic) should dispatch to superior');

%% plus: basic obj first — should STILL dispatch to superior
r = plus(a, s);
assert(strcmp(r, 'superior'), 'plus(basic, superior) should dispatch to superior');

%% operator syntax
r = s + a;
assert(strcmp(r, 'superior'), 's + a should dispatch to superior');

r = a + s;
assert(strcmp(r, 'superior'), 'a + s should dispatch to superior');

%% minus: tests internal redispatch (minus calls plus)
r = s - a;
assert(strcmp(r, 'superior'), 's - a should dispatch to superior minus->plus');

r = a - s;
assert(strcmp(r, 'superior'), 'a - s should dispatch to superior minus->plus');

%% mtimes
r = s * a;
assert(strcmp(r, 'superior'), 's * a should dispatch to superior');

r = a * s;
assert(strcmp(r, 'superior'), 'a * s should dispatch to superior');

%% both superior — should still work
r = s + s;
assert(strcmp(r, 'superior'), 's + s should dispatch to superior');

%% both basic — should still work
r = a + a;
assert(strcmp(r, 'basic'), 'a + a should dispatch to basic');

disp('SUCCESS')
