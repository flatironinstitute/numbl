% Test that @Class/private/ functions with local subfunctions work
% when local function args have unknown types (from class property access)
obj = FolderClass(5);
assert(obj.val == 11);  % privateHelper(obj) -> localDouble(obj.val) + 1 = 10 + 1 = 11
disp('SUCCESS')
