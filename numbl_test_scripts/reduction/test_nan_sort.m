% Test NaN handling in sort
% MATLAB puts NaN at the end for ascending, at the beginning for descending.

% Test 1: sort ascending with NaN
r = sort([3 NaN 1 2]);
assert(r(1) == 1, 'sorted(1) should be 1');
assert(r(2) == 2, 'sorted(2) should be 2');
assert(r(3) == 3, 'sorted(3) should be 3');
assert(isnan(r(4)), 'sorted(4) should be NaN');

% Test 2: sort descending with NaN
r2 = sort([3 NaN 1 2], 'descend');
assert(isnan(r2(1)), 'desc sorted(1) should be NaN');
assert(r2(2) == 3, 'desc sorted(2) should be 3');
assert(r2(3) == 2, 'desc sorted(3) should be 2');
assert(r2(4) == 1, 'desc sorted(4) should be 1');

% Test 3: sort with multiple NaN
r3 = sort([NaN 1 NaN 2]);
assert(r3(1) == 1, 'multiple NaN: sorted(1) should be 1');
assert(r3(2) == 2, 'multiple NaN: sorted(2) should be 2');
assert(isnan(r3(3)), 'multiple NaN: sorted(3) should be NaN');
assert(isnan(r3(4)), 'multiple NaN: sorted(4) should be NaN');

% Test 4: sort all NaN
r4 = sort([NaN NaN NaN]);
assert(isnan(r4(1)), 'all NaN sort(1)');
assert(isnan(r4(2)), 'all NaN sort(2)');
assert(isnan(r4(3)), 'all NaN sort(3)');

% Test 5: sort with index output
[r5, i5] = sort([3 NaN 1 2]);
assert(r5(1) == 1, 'sort idx: value(1) should be 1');
assert(i5(1) == 3, 'sort idx: index(1) should be 3');
assert(r5(3) == 3, 'sort idx: value(3) should be 3');
assert(i5(3) == 1, 'sort idx: index(3) should be 1');
assert(isnan(r5(4)), 'sort idx: value(4) should be NaN');
assert(i5(4) == 2, 'sort idx: index(4) should be 2');

disp('SUCCESS');
