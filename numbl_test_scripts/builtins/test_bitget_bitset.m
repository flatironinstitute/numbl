% Test bitget and bitset.

% --- bitget: scalar A, scalar bit ---
assert(bitget(5, 1) == 1, '5 bit 1');
assert(bitget(5, 2) == 0, '5 bit 2');
assert(bitget(5, 3) == 1, '5 bit 3');
assert(bitget(5, 4) == 0, '5 bit 4');
assert(bitget(0, 1) == 0, '0 bit 1');
assert(bitget(255, 8) == 1, '255 bit 8');

% --- bitget: vector A, scalar bit ---
r = bitget([1 2 3 4 5 6 7 8], 1);
assert(isequal(r, [1 0 1 0 1 0 1 0]), 'vec A, scalar bit (LSB)');

r = bitget([1 2 3 4 5 6 7 8], 2);
assert(isequal(r, [0 1 1 0 0 1 1 0]), 'vec A, scalar bit (bit 2)');

% --- bitget: scalar A, vector bits (chunkie/FLAM usage) ---
r = bitget(5, 1:4);
assert(isequal(r, [1 0 1 0]), 'scalar A, vec bits');
r = bitget(255, 1:8);
assert(isequal(r, [1 1 1 1 1 1 1 1]), 'FF bits');

% --- bitget: same-size A and bits ---
r = bitget([1 2 4 8], [1 2 3 4]);
assert(isequal(r, [1 1 1 1]), 'matched A and bits');

% --- bitset: default value = 1 ---
assert(bitset(0, 1) == 1, 'set bit 1 on 0');
assert(bitset(0, 3) == 4, 'set bit 3 on 0');
assert(bitset(1, 3) == 5, 'set bit 3 on 1');

% --- bitset with explicit value ---
assert(bitset(15, 1, 0) == 14, 'clear bit 1');
assert(bitset(15, 4, 0) == 7, 'clear bit 4');
assert(bitset(0, 1, 1) == 1, 'set bit 1 explicit');

% --- bitset: vector A ---
r = bitset([0 0 0 0], [1 2 3 4]);
assert(isequal(r, [1 2 4 8]), 'vec A');

% --- bitset with clear, vector A ---
r = bitset([15 15 15 15], [1 2 3 4], 0);
assert(isequal(r, [14 13 11 7]), 'clear vec');

disp('SUCCESS');
