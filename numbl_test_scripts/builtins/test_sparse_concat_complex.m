% Test sparse concatenation with complex scalars

%% horzcat sparse with complex tensor column
S = sparse([1 0; 0 2]);
R = [S, [3+4i; 5+6i]];
assert(issparse(R));
expected = [1 0 3+4i; 0 2 5+6i];
assert(isequal(full(R), expected));

%% horzcat sparse with bare complex scalar
S2 = sparse(1);
R2 = [S2, 1+2i];
assert(issparse(R2));
assert(isequal(full(R2), [1, 1+2i]));

%% vertcat sparse with bare complex scalar
R3 = [S2; 3+4i];
assert(issparse(R3));
assert(isequal(full(R3), [1; 3+4i]));

disp('SUCCESS')
