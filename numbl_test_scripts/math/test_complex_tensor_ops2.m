% Test more complex tensor operations that lose imaginary data

% Bug 1: unique on complex arrays drops imaginary parts
r1 = unique([3+1i, 1+2i, 3+1i, 2+0i]);
assert(length(r1) == 3, 'unique complex length');
assert(any(r1 == 1+2i), 'unique preserves 1+2i');
assert(any(r1 == 2), 'unique preserves 2');
assert(any(r1 == 3+1i), 'unique preserves 3+1i');

% Bug 2: diff on complex tensors drops imaginary parts
r2 = diff([1+1i, 3+2i, 6+4i]);
assert(r2(1) == 2+1i, 'diff complex (1)');
assert(r2(2) == 3+2i, 'diff complex (2)');

% Bug 3: repmat with complex scalar input
r3 = repmat(1+2i, 1, 3);
assert(isequal(size(r3), [1 3]), 'repmat complex size');
assert(r3(1) == 1+2i, 'repmat complex val 1');
assert(r3(2) == 1+2i, 'repmat complex val 2');
assert(r3(3) == 1+2i, 'repmat complex val 3');

disp('SUCCESS');
