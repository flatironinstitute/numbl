% Test overloaded subsref on class instances

% Test 1: single dot access via subsref
p = SubsrefPrefs_();
assert(p.alpha == 10);
assert(p.beta == 20);

% Test 2: chained dot access via subsref
assert(p.techPrefs.epsilon == 0.001);
assert(p.techPrefs.maxiter == 500);

disp('SUCCESS')
