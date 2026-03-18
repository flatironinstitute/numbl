% Test that cell arrays preserve sparsity of their elements
AA = cell(1, 1);
AA{1, 1} = sparse(eye(3));
assert(issparse(AA{1, 1}));
assert(isequal(AA{1, 1}, sparse(eye(3))));

% Also test with a cell vector
B = cell(1, 3);
B{1} = sparse([1 0; 0 2]);
B{2} = 42;
B{3} = sparse([0; 5; 0]);
assert(issparse(B{1}));
assert(~issparse(B{2}));
assert(issparse(B{3}));
assert(isequal(B{1}, sparse([1 0; 0 2])));
assert(isequal(B{3}, sparse([0; 5; 0])));

disp('SUCCESS')
