%% setdiff returns sorted values not in B
[c, ia] = setdiff([1 2 3 4], [1 4]);
assert(isequal(c, [2 3]));
assert(isequal(ia, [2 3]'));

%% c = A(ia) relationship holds
A = [10 20 30 40 50];
B = [20 40];
[c2, ia2] = setdiff(A, B);
assert(isequal(c2, [10 30 50]));
assert(isequal(c2, A(ia2)));

%% setdiff with no common elements
[c3, ia3] = setdiff([5 6], [1 2]);
assert(isequal(c3, [5 6]));
assert(isequal(ia3, [1 2]'));

%% setdiff with all common elements returns empty
[c4, ia4] = setdiff([1 2], [1 2 3]);
assert(isempty(c4));
assert(isempty(ia4));

%% setdiff single output (no ia)
c5 = setdiff([3 1 4 1 5], [1 5]);
assert(isequal(c5, [3 4]));

disp('SUCCESS')
