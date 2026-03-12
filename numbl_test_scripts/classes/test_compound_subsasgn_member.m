% Test that V(i).field = rhs uses built-in index store for class instances.
% In MATLAB, V(i).field = rhs calls subsasgn with a compound index.
% The numbl codegen decomposes it into get/set/store-back steps.
% The store-back must NOT re-trigger subsasgn on the class.

obj = CompoundAssignObj();
obj.data = [1, 2, 3];

% This is the compound pattern: obj(1).data = newval
obj(1).data = [7, 8, 9];

if isequal(obj.data, [7, 8, 9])
    disp('SUCCESS');
else
    disp('FAILURE: compound V(i).field assignment did not work');
end
