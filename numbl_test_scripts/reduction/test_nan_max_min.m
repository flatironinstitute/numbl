% Test NaN handling in max and min
% MATLAB skips NaN values and only returns NaN if ALL values are NaN.

% Test 1: max skips NaN in vector
assert(max([1 NaN 3]) == 3, 'max should skip NaN');

% Test 2: max returns NaN when all values are NaN
assert(isnan(max([NaN NaN])), 'max of all NaN should be NaN');

% Test 3: max with NaN and index output
[v, i] = max([NaN 2 1]);
assert(v == 2, 'max value should skip NaN');
assert(i == 2, 'max index should point to non-NaN winner');

% Test 4: max with NaN at end
[v, i] = max([1 NaN 3]);
assert(v == 3, 'max should skip interior NaN');
assert(i == 3, 'max index should be 3');

% Test 5: min skips NaN
assert(min([1 NaN 3]) == 1, 'min should skip NaN');

% Test 6: min returns NaN when all NaN
assert(isnan(min([NaN NaN])), 'min of all NaN should be NaN');

% Test 7: min with index output
[v, i] = min([1 NaN 3]);
assert(v == 1, 'min value should skip NaN');
assert(i == 1, 'min index should be 1');

% Test 8: max along dim 1 with NaN
A = [1 NaN; NaN 4];
r = max(A, [], 1);
assert(r(1) == 1, 'max along dim 1, col 1 should skip NaN');
assert(r(2) == 4, 'max along dim 1, col 2 should skip NaN');

% Test 9: max along dim 2 with NaN
r2 = max(A, [], 2);
assert(r2(1) == 1, 'max along dim 2, row 1 should skip NaN');
assert(r2(2) == 4, 'max along dim 2, row 2 should skip NaN');

% Test 10: min along dim 1 with NaN
r3 = min(A, [], 1);
assert(r3(1) == 1, 'min along dim 1, col 1 should skip NaN');
assert(r3(2) == 4, 'min along dim 1, col 2 should skip NaN');

% Test 11: all-NaN column along dim
B = [NaN NaN; NaN 1];
r4 = max(B, [], 1);
assert(isnan(r4(1)), 'all-NaN column max should be NaN');
assert(r4(2) == 1, 'mixed column max should skip NaN');

% Test 12: max of matrix (no dim specified) reduces along dim 1
C = [NaN 3; 1 NaN];
r5 = max(C);
assert(r5(1) == 1, 'max matrix col 1 should skip NaN');
assert(r5(2) == 3, 'max matrix col 2 should skip NaN');

% Test 13: min of matrix (no dim specified)
r6 = min(C);
assert(r6(1) == 1, 'min matrix col 1 should skip NaN');
assert(r6(2) == 3, 'min matrix col 2 should skip NaN');

% Test 14: single NaN element
assert(isnan(max(NaN)), 'max of scalar NaN should be NaN');
assert(isnan(min(NaN)), 'min of scalar NaN should be NaN');

% Test 15: NaN with negative values
assert(max([NaN -5 -1]) == -1, 'max should skip NaN with negatives');
assert(min([NaN -5 -1]) == -5, 'min should skip NaN with negatives');

disp('SUCCESS');
