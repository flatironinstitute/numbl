% Test regexp 'split' output mode

parts = regexp('2.2 0 8', '\s+', 'split');
assert(iscell(parts));
assert(numel(parts) == 3);
assert(strcmp(parts{1}, '2.2'));
assert(strcmp(parts{2}, '0'));
assert(strcmp(parts{3}, '8'));

% leading/trailing delimiters produce empty pieces
p2 = regexp(',a,,b,', ',', 'split');
assert(numel(p2) == 5);
assert(isempty(p2{1}) && isempty(p2{3}) && isempty(p2{5}));
assert(strcmp(p2{2}, 'a') && strcmp(p2{4}, 'b'));

% no match: whole string in one piece
p3 = regexp('abc', 'x', 'split');
assert(numel(p3) == 1 && strcmp(p3{1}, 'abc'));

% 'split' combined with 'once': split around the first match only
p4 = regexp('a-b-c', '-', 'split', 'once');
assert(numel(p4) == 2 && strcmp(p4{1}, 'a') && strcmp(p4{2}, 'b-c'));

disp('SUCCESS')
