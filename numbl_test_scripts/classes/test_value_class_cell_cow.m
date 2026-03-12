% Test copy-on-write semantics for value classes with cell array properties
% containing class instances. In MATLAB, modifying a method parameter
% (value class) should not affect the caller's original object.

% Helper classes defined in same directory:
% CellHolder.m - has cell array 'funs' property, uminus negates each element
% InnerObj.m - has 'coeffs' property, uminus negates coeffs

% Basic test: single negation
inner1 = InnerObj([1 2 3]);
inner2 = InnerObj([4 5 6]);
holder = CellHolder({inner1, inner2});

neg = -holder;
assert(isequal(holder.funs{1}.coeffs, [1 2 3]));
assert(isequal(holder.funs{2}.coeffs, [4 5 6]));
assert(isequal(neg.funs{1}.coeffs, [-1 -2 -3]));
assert(isequal(neg.funs{2}.coeffs, [-4 -5 -6]));

% Repeated negation should not corrupt original
neg2 = -holder;
assert(isequal(holder.funs{1}.coeffs, [1 2 3]));
assert(isequal(holder.funs{2}.coeffs, [4 5 6]));

neg3 = -holder;
assert(isequal(holder.funs{1}.coeffs, [1 2 3]));
assert(isequal(holder.funs{2}.coeffs, [4 5 6]));

disp('SUCCESS');
