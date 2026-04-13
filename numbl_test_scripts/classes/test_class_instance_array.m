% Test default class instance array (no horzcat overload)

a = SimpleObjNoHorzcat_(1);
b = SimpleObjNoHorzcat_(2);
c = SimpleObjNoHorzcat_(3);

% Test 1: [a b] creates a 1x2 array
arr = [a b];
assert(numel(arr) == 2, 'numel should be 2');
assert(length(arr) == 2, 'length should be 2');
assert(~isempty(arr), 'should not be empty');
assert(isa(arr, 'SimpleObjNoHorzcat_'), 'isa should match class');

% Test 2: indexing
first = arr(1);
assert(first.val == 1, 'arr(1).val should be 1');
second = arr(2);
assert(second.val == 2, 'arr(2).val should be 2');

% Test 3: grow in loop
list = a;
for j = 1:3
    list = [list SimpleObjNoHorzcat_(j*10)];
end
assert(numel(list) == 4, 'numel should be 4 after loop');
assert(list(1).val == 1, 'list(1).val should be 1');
assert(list(2).val == 10, 'list(2).val should be 10');
assert(list(3).val == 20, 'list(3).val should be 20');
assert(list(4).val == 30, 'list(4).val should be 30');

% Test 4: size
s = size(arr);
assert(isequal(s, [1 2]), 'size should be [1 2]');

disp('SUCCESS');
