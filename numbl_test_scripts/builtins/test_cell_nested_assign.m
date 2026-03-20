% Test that curly-brace assignment preserves cell RHS
% c{i} = {val} should store the cell {val}, not unwrap it

c = cell(1, 2);
c{1} = {42};
c{2} = {1, 2, 3};

% c{1} should be a cell containing 42
assert(iscell(c{1}), 'c{1} should be a cell');
assert(c{1}{1} == 42, 'c{1}{1} should be 42');

% c{2} should be a cell containing 1,2,3
assert(iscell(c{2}), 'c{2} should be a cell');
assert(numel(c{2}) == 3, 'c{2} should have 3 elements');
assert(c{2}{1} == 1);
assert(c{2}{2} == 2);
assert(c{2}{3} == 3);

% Paren assignment should still unwrap: c(i) = {val} stores val directly
c2 = cell(1, 2);
c2(1) = {99};
c2(2) = {[1 2 3]};
assert(~iscell(c2{1}), 'c2(1) = {99} should store 99, not {99}');
assert(c2{1} == 99);
assert(isequal(c2{2}, [1 2 3]));

% Regression: vertcat result assigned into cell should stay as cell
tmp = cell(1, 1);
tmp{1} = 42;
x = tmp(:, 1);     % paren indexing on cell -> cell
y = vertcat(x);    % single-arg vertcat -> same cell
out = cell(1, 1);
out{1} = y;         % curly-brace assign cell into cell
assert(iscell(out{1}), 'out{1} should be a cell after c{1} = cell_value');

% horzcat of cell-of-cells then cell indexing
out2 = horzcat(out{:});
assert(iscell(out2), 'horzcat of cells should return a cell');
assert(out2{1} == 42, 'out2{1} should be 42');

disp('SUCCESS');
