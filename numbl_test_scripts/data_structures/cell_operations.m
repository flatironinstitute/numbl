% Test cell array operations

% Create cell array with mixed types
c = {1, 'hello', [1, 2, 3]};
assert(c{1} == 1);
assert(strcmp(c{2}, 'hello'));
assert(c{3}(2) == 2);

% Nested cell arrays
nc = {{1, 2}, {3, 4}};
assert(nc{1}{2} == 2);
assert(nc{2}{1} == 3);

% iscell
assert(iscell(c));
assert(~iscell([1, 2, 3]));

% length and numel on cell
assert(length(c) == 3);
assert(numel(c) == 3);

% Iterate over cell with for loop
words = {'apple', 'banana', 'cherry'};
count = 0;
for i = 1:length(words)
  count = count + length(words{i});
end
% apple=5, banana=6, cherry=6 → 17
assert(count == 17);

disp('SUCCESS')
