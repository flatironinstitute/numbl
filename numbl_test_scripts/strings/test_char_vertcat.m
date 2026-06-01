% Vertical concatenation of char rows must build a 2-D char array.
% numbl's vertcat had no char branch (only horzcat did), so ['ab';'cd']
% threw "Cannot concatenate char into matrix".

x = ['ab'; 'cd'];
assert(ischar(x), 'result should be char');
assert(isequal(size(x), [2 2]), 'char vertcat size should be 2x2');
assert(strcmp(x(1, :), 'ab'), 'row 1 should be ''ab''');
assert(strcmp(x(2, :), 'cd'), 'row 2 should be ''cd''');

% three rows
y = ['abc'; 'def'; 'ghi'];
assert(isequal(size(y), [3 3]), '3-row char vertcat size');
assert(strcmp(y(2, :), 'def'), '3-row middle row');

% mismatched widths must error (dimension mismatch), not silently succeed
ok = false;
try
    z = ['ab'; 'cde']; %#ok<NASGU>
catch
    ok = true;
end
assert(ok, 'mismatched char row widths should error');

disp('SUCCESS');
