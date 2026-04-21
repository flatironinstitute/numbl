% C-JIT parity gap #20: RangeSliceRead loses row-vector orientation.
%
% `y = v(a:b)` should return a row when v is a row and a column when v
% is a column (MATLAB semantics, JS-JIT behavior). The C-JIT's
% RangeSliceRead-to-dynamic-output path hard-coded the result as a
% column vector (d0 = len, d1 = 1), dropping the source's isRow flag.
%
% Expected disp output (must match across all runs):
%   numbl --opt 1 run <this>  -> 1\n4\n4\n1\nSUCCESS
%   numbl --opt 2 run <this>  -> 1\n4\n4\n1\nSUCCESS
%   matlab -batch parity20_range_slice_orientation -> 1\n4\n4\n1\nSUCCESS

% 1) Row-vector source: v(2:5) must stay a row.
v_row = 1:10;
y_row = slice_it(v_row, 2, 5);
disp(size(y_row, 1))    % 1
disp(size(y_row, 2))    % 4
assert(isequal(size(y_row), [1 4]), 'row slice orientation');

% 2) Column-vector source: v(2:5) must stay a column.
v_col = (1:10)';
y_col = slice_it(v_col, 2, 5);
disp(size(y_col, 1))    % 4
disp(size(y_col, 2))    % 1
assert(isequal(size(y_col), [4 1]), 'col slice orientation');

disp('SUCCESS')

function r = slice_it(v, a, b)
    r = v(a:b);
end
