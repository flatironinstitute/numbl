% Test that calling a static method via an instance does NOT prepend the
% instance as first argument. In MATLAB, obj.staticMethod(a, b) passes
% only (a, b) to the static method, using obj only for dispatch.

obj = TechLike(42);
pref.value = 10;

% Call static method via instance — should get refine(3, pref) => 13
r = obj.refine(3, pref);
assert(r == 13, sprintf('Expected 13, got %g', r));

% Also call through an instance method that delegates to obj.refine
r2 = obj.callRefine(5, pref);
assert(r2 == 15, sprintf('Expected 15, got %g', r2));

% Call static method via class name — should also work
r3 = TechLike.refine(7, pref);
assert(r3 == 17, sprintf('Expected 17, got %g', r3));

disp('SUCCESS');
