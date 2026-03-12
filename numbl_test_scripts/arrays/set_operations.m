% Test set operations: intersect, union, setdiff, ismember

%% intersect - elements common to both, sorted
c = intersect([3, 1, 4, 1, 5], [1, 5, 9]);
assert(length(c) == 2);
assert(c(1) == 1);
assert(c(2) == 5);

% No common elements
c2 = intersect([1, 2], [3, 4]);
assert(isempty(c2));

% All common
c3 = intersect([2, 1, 3], [3, 2, 1]);
assert(length(c3) == 3);
assert(c3(1) == 1);
assert(c3(2) == 2);
assert(c3(3) == 3);

%% union - combined unique elements, sorted
u = union([3, 1, 4], [1, 5, 9]);
assert(length(u) == 5);
assert(u(1) == 1);
assert(u(2) == 3);
assert(u(3) == 4);
assert(u(4) == 5);
assert(u(5) == 9);

% Duplicates within inputs
u2 = union([1, 1, 2], [2, 3, 3]);
assert(length(u2) == 3);
assert(u2(1) == 1);
assert(u2(2) == 2);
assert(u2(3) == 3);

%% setdiff - elements in A but not in B, sorted
d = setdiff([3, 1, 4, 1, 5], [1, 5]);
assert(length(d) == 2);
assert(d(1) == 3);
assert(d(2) == 4);

% Nothing removed
d2 = setdiff([1, 2, 3], [4, 5]);
assert(length(d2) == 3);

% All removed
d3 = setdiff([1, 2], [1, 2, 3]);
assert(isempty(d3));

%% ismember - logical array: is each element of A in B?
tf = ismember([1, 2, 3, 4, 5], [2, 4, 6]);
assert(tf(1) == false);
assert(tf(2) == true);
assert(tf(3) == false);
assert(tf(4) == true);
assert(tf(5) == false);

% Scalar
assert(ismember(3, [1, 2, 3]) == true);
assert(ismember(4, [1, 2, 3]) == false);

% Column vector
tf2 = ismember([1; 2; 3], [2, 3]);
assert(tf2(1) == false);
assert(tf2(2) == true);
assert(tf2(3) == true);

disp('SUCCESS');
