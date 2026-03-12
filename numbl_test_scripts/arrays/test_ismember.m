%% Test ismember - single output (logical)
tf = ismember([1 3 5], [2 3 4 5]);
assert(isequal(tf, logical([0 1 1])));

%% Test ismember - two outputs: [tf, loc]
[tf2, loc2] = ismember([1 3 5 7], [2 3 4 5]);
assert(isequal(tf2, logical([0 1 1 0])));
assert(isequal(loc2, [0 2 4 0]));

%% Test ismember - scalar in vector
[tf3, loc3] = ismember(3, [1 2 3 4 5]);
assert(tf3 == true);
assert(loc3 == 3);

%% Test ismember - scalar not in vector
[tf4, loc4] = ismember(99, [1 2 3]);
assert(tf4 == false);
assert(loc4 == 0);

%% Test ismember - duplicate values in B (returns first/lowest index)
[tf5, loc5] = ismember([1 2], [2 1 2 1]);
assert(isequal(tf5, logical([1 1])));
assert(isequal(loc5, [2 1]));

%% Test ismember - column vector input
[tf6, loc6] = ismember([1; 3; 5], [5 3 1]);
assert(isequal(tf6, logical([1; 1; 1])));
assert(isequal(loc6, [3; 2; 1]));

disp('SUCCESS')
