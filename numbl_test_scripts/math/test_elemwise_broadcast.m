% Two-argument elementwise builtins must broadcast like .* / + / power do.
% numbl's interpreter implementations of atan2/mod/rem/hypot/nthroot only
% checked equal element COUNT, then paired 1:1 and returned the first
% operand's shape -- giving wrong results for equal-count/different-shape
% operands and crashing on broadcast-compatible/different-count operands.

% mod / rem: 1x3 vs 3x1 -> 3x3 broadcast
assert(isequal(mod([1 2 3], [10; 20; 30]), [1 2 3; 1 2 3; 1 2 3]), 'mod broadcast');
assert(isequal(rem([1 2 3], [10; 20; 30]), [1 2 3; 1 2 3; 1 2 3]), 'rem broadcast');

% atan2: 1x2 vs 3x1 -> 3x2 (different element counts, must NOT crash)
A = atan2([1 2], [10; 20; 30]);
assert(isequal(size(A), [3 2]), 'atan2 broadcast size');
assert(abs(A(2, 1) - atan2(1, 20)) < 1e-12, 'atan2 broadcast value');

% hypot: equal count, different shape (1x2 vs 2x1 -> 2x2)
H = hypot([3 4], [3; 4]);
assert(isequal(H, [hypot(3, 3) hypot(4, 3); hypot(3, 4) hypot(4, 4)]), 'hypot broadcast');

% scalar with vector still works
assert(isequal(mod(10, [3 4 7]), [1 2 3]), 'mod scalar-vector');

disp('SUCCESS');
