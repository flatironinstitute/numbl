% Regression: `tmp = obj(1)` on a value class without subsref must
% return an independent value, not alias `obj`. If the runtime returns
% `obj` itself (the same reference), a subsequent `tmp = ...` rebind
% disposes obj's buffers via the Assign dispose path and corrupts obj.

obj = ScalarHolder_([1 2 3 4]);

% Default paren-indexing (no subsref defined): obj(1) used to return
% obj itself. With the fix, it returns a deep clone.
tmp = obj(1);

% Rebind tmp via a FuncCall (owned RHS — Assign dispose path will fire
% on the previous tmp). Without the fix, this dispose poisons obj's
% data with NaN.
tmp = ScalarHolder_([9 9 9 9]);

% obj must still hold its original data.
assert(obj.data(1) == 1, 'obj.data(1) should be 1');
assert(obj.data(2) == 2, 'obj.data(2) should be 2');
assert(obj.data(3) == 3, 'obj.data(3) should be 3');
assert(obj.data(4) == 4, 'obj.data(4) should be 4');

% Repeat with full-slice index: obj(:) also takes the same default path.
obj2 = ScalarHolder_([10 20 30]);
slice = obj2(:);
slice = ScalarHolder_([0 0 0]);
assert(obj2.data(1) == 10, 'obj2.data(1) should be 10');
assert(obj2.data(2) == 20, 'obj2.data(2) should be 20');
assert(obj2.data(3) == 30, 'obj2.data(3) should be 30');

disp('SUCCESS');
