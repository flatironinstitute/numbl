%% find with 'last' and count=1
x = [1 2 3 4 5];
n = find(x > 2, 1, 'last');
assert(n == 5);

%% find with 'last' and count=2
n2 = find(x > 2, 2, 'last');
assert(isequal(n2, [4 5]));

%% find with 'first' and count=1
n3 = find(x > 2, 1, 'first');
assert(n3 == 3);

%% find with 'first' and count=2
n4 = find(x > 2, 2, 'first');
assert(isequal(n4, [3 4]));

%% find with 'last' and count=1, only one match
y = [0 0 1 0 0];
n5 = find(y, 1, 'last');
assert(n5 == 3);

%% find with 'last' and count=1, all match
z = [1 1 1 1 1];
n6 = find(z, 1, 'last');
assert(n6 == 5);

disp('SUCCESS')
