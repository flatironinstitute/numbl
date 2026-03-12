% Test that V(i).field{k} = rhs inside a class method uses builtin indexing
% for the V(i) part, not the class's overloaded subsasgn.
% This matches MATLAB behavior for compound assignments.

obj = ChainedCellAsgnObj_(3);
assert(obj.items{1} == 10);
assert(obj.items{2} == 20);
assert(obj.items{3} == 30);

obj = negateItems(obj);
assert(obj.items{1} == -10);
assert(obj.items{2} == -20);
assert(obj.items{3} == -30);

disp('SUCCESS');
