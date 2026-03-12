% Test find, ceil, floor, round with complex numbers

%% find with complex tensors
idx = find([0, 1+2i, 0, 3i]);
assert(isequal(idx, [2, 4]), 'find should detect nonzero imag parts');

idx2 = find([0+0i, 0+1i, 1+0i]);
assert(isequal(idx2, [2, 3]), 'find should detect pure imaginary as nonzero');

%% ceil with complex
assert(ceil(1.2 + 3.7i) == 2 + 4i, 'ceil should work on complex');
assert(isequal(ceil([-1.5 + 2.3i, 0.1 - 0.9i]), [-1 + 3i, 1 + 0i]), 'ceil should work on complex tensors');

%% floor with complex
assert(floor(1.7 + 3.2i) == 1 + 3i, 'floor should work on complex');
assert(isequal(floor([-1.5 + 2.3i, 0.1 - 0.9i]), [-2 + 2i, 0 - 1i]), 'floor should work on complex tensors');

%% round with complex
assert(round(1.5 + 2.5i) == 2 + 3i, 'round should work on complex');

disp('SUCCESS');
