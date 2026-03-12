%% Property access works
obj = PropMethod_(42);
assert(obj.value == 42);

%% Method call with same name as property works
out = value(obj);
assert(out == 42);

%% Method call with extra arg
out2 = value(obj, 'double');
assert(out2 == 84);

%% Dot-syntax with indexing should access property, not call method
%% obj.domain(1) should return first element of domain property [1 2 3 4 5]
d1 = obj.domain(1);
assert(d1 == 1);

%% obj.domain(3) should return third element
d3 = obj.domain(3);
assert(d3 == 3);

%% obj.domain should return the full property
d = obj.domain;
assert(isequal(d, [1 2 3 4 5]));

%% Function-call syntax should call the method
out3 = domain(obj);
assert(isequal(out3, [1 2 3 4 5]));

%% Function-call syntax with flag calls the method
out4 = domain(obj, 'ends');
assert(isequal(out4, [1 5]));

disp('SUCCESS')
