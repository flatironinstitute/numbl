%% Create a class that inherits from double
d = MyDouble_([1 2 3], 'test');
assert(strcmp(d.label, 'test'));

%% Extract underlying double data
data = double(d);
assert(isequal(data, [1 2 3]));

%% getData method also extracts the data
data2 = d.getData();
assert(isequal(data2, [1 2 3]));

%% isnumeric should return true for a double subclass
assert(isnumeric(d));

%% Empty construction
e = MyDouble_();
data3 = double(e);
assert(isempty(data3));

disp('SUCCESS')
