% cell2mat with char contents concatenates into a char array.

% Row cell of char rows -> single char row
s = cell2mat({'ab', 'cde', 'f'});
assert(ischar(s));
assert(strcmp(s, 'abcdef'));

% The cellfun/append-space idiom
u = strtrim(cell2mat(cellfun(@(x) [x ' '], {'excitation', 'refocusing'}, ...
    'UniformOutput', false)));
assert(strcmp(u, 'excitation refocusing'));

% Column cell of equal-width char rows -> char matrix
mchar = cell2mat({'abc'; 'def'});
assert(ischar(mchar));
assert(isequal(size(mchar), [2 3]));
assert(strcmp(mchar(1, :), 'abc'));
assert(strcmp(mchar(2, :), 'def'));

% Single-element cell
assert(strcmp(cell2mat({'xyz'}), 'xyz'));

% Numeric contents still work
assert(isequal(cell2mat({[1 2], [3]}), [1 2 3]));

disp('SUCCESS');
