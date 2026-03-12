% Test isscalar, any, all with complex numbers

%% isscalar with complex numbers
z = 1 + 2i;
assert(isscalar(z), 'isscalar should be true for complex scalar');
assert(isscalar(3i), 'isscalar should be true for pure imaginary');
assert(isscalar(0 + 0i), 'isscalar should be true for zero complex');

%% any with complex scalars
assert(any(1 + 2i), 'any should be true for nonzero complex');
assert(any(3i), 'any should be true for pure imaginary');
assert(~any(0), 'any should be false for zero');

%% any with complex vectors
v = [0 0i 0];
assert(~any(v), 'any should be false for all-zero vector');
v2 = [0 1i 0];
assert(any(v2), 'any should be true when imag part is nonzero');

%% all with complex scalars
assert(all(1 + 2i), 'all should be true for nonzero complex');
assert(all(3i), 'all should be true for pure imaginary');
assert(~all(0), 'all should be false for zero');

%% all with complex vectors
v3 = [1+2i 3i 4];
assert(all(v3), 'all should be true when all elements nonzero');
v4 = [1+2i 0 3i];
assert(~all(v4), 'all should be false when any element is zero');

disp('SUCCESS');
