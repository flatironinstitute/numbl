% Test chained member + cell index + member assignment
% e.g., f.cols.funs{1}.onefun = value

% Build a nested struct with a cell array
s = struct();
s.cols = struct();
s.cols.funs = {struct('onefun', 0)};

% Assign through the chain
s.cols.funs{1}.onefun = 42;

assert(s.cols.funs{1}.onefun == 42);

% Also test with regular index in the chain
t = struct();
t.data = struct();
t.data.items = {struct('val', 10), struct('val', 20)};
t.data.items{2}.val = 99;
assert(t.data.items{2}.val == 99);
assert(t.data.items{1}.val == 10);

disp('SUCCESS');
