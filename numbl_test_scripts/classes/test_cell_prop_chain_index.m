% obj.prop{k}(j) where k is a 1x1 tensor (e.g. from a two-output find)
% must extract the cell element and index into it — both from outside the
% class and from inside a method.

m = PulseqPatterns_();
m.c = {[1 2 3], [4 5 6], [7 8 9], [10 11 12]};

% From outside the class
d = [1 2 0 5];
[~, k] = find(d > 0, 1, 'last');
assert(m.c{k}(3) == 12);

% Plain scalar index too
assert(m.c{2}(1) == 4);

% From inside a method (the original failing context)
assert(m.lastPositive([1 2 0 5], 3) == 12);
assert(m.lastPositive([1 0 0 0], 2) == 2);

disp('SUCCESS');
