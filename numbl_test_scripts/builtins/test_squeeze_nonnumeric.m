% squeeze accepts any array type (not just numeric), like MATLAB. For
% types that are at most 2-D it is a no-op; for N-D numerics it drops
% singleton dimensions.

% --- scalar struct: unchanged ---
s.a = 1; s.b = 2;
t = squeeze(s);
assert(isstruct(t) && ndims(t) == 2 && t.a == 1 && t.b == 2, 'scalar struct');

% --- struct array: unchanged ---
sa = struct('x', {1, 2, 3});
assert(numel(squeeze(sa)) == 3, 'struct array');

% --- cell array: unchanged when 2-D ---
c = {1, 'two', [3 3]};
sc = squeeze(c);
assert(iscell(sc) && numel(sc) == 3 && strcmp(sc{2}, 'two'), 'cell array');

% --- char row: unchanged ---
assert(strcmp(squeeze('hello'), 'hello'), 'char row');

% --- numeric: still drops singleton dims ---
assert(isequal(size(squeeze(ones(1, 3))), [1 3]), '2-D numeric no-op');
assert(isequal(size(squeeze(ones(2, 1, 3))), [2 3]), '3-D numeric squeezed');
assert(isequal(size(squeeze(ones(1, 1, 4))), [4 1]), 'squeeze to column');

disp('SUCCESS');
