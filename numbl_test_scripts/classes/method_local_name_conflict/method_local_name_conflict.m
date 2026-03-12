% Test: two method files each define a local function with the same name.
% compute_a uses localHelper that adds 10; compute_b uses localHelper that multiplies by 3.
% Each method must call its OWN local helper, not the other's.

obj = LocalConflict_(4);

% compute_a uses localHelper: value + 10 = 14
a = compute_a(obj);
assert(a == 14, sprintf('compute_a: expected 14, got %g', a));

% compute_b uses its own localHelper: value * 3 = 12
b = compute_b(obj);
assert(b == 12, sprintf('compute_b: expected 12, got %g', b));

disp('SUCCESS')
