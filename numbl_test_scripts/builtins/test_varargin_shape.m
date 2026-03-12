% Test that varargin has shape [1, n] (row cell), not [n, 1] (column cell)

% Test 1: varargin should be 1xN row cell
c = get_varargin(10, 20, 30);
assert(isequal(size(c), [1, 3]), 'varargin should be 1x3 row cell');
assert(isequal(c{1}, 10), 'first element should be 10');
assert(isequal(c{3}, 30), 'third element should be 30');

% Test 2: cellfun on varargin should preserve row shape
lens = cellfun(@(x) x, get_varargin(1, 2, 3));
assert(isequal(size(lens), [1, 3]), 'cellfun on varargin should be 1x3');

% Test 3: varargin with 2 elements
c2 = get_varargin('a', 'b');
assert(isequal(size(c2), [1, 2]), 'varargin should be 1x2 row cell');

disp('SUCCESS')

function c = get_varargin(varargin)
    c = varargin;
end
