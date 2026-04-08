% Test vertical and horizontal concatenation of struct arrays.  numbl
% doesn't track struct-array shape, so results are compared by
% traversing with numel and field access.

% --- Basic vertcat of 1D struct arrays ---
s = struct('a', {1, 2}, 'b', {'x', 'y'});
t = struct('a', {3, 4}, 'b', {'u', 'v'});

r = [s; t];
assert(numel(r) == 4, sprintf('vertcat numel = %d', numel(r)));
vals = [r.a];
assert(any(vals == 1) && any(vals == 2) && any(vals == 3) && any(vals == 4), ...
    'vertcat values missing');

% --- horzcat of 1D struct arrays ---
rh = [s, t];
assert(numel(rh) == 4, sprintf('horzcat numel = %d', numel(rh)));
valsH = [rh.a];
assert(sum(valsH) == 10, 'horzcat value sum');

% --- Struct array + single struct ---
single = struct('a', 99, 'b', 'z');
r2 = [s, single];
assert(numel(r2) == 3, 'single append numel');
assert(r2(end).a == 99, 'single append value');

% --- Chunkie-style growth loop: start with empty cell-init struct, append ---
e = cell(0, 1);
T = struct('v', e);
assert(numel(T) == 0, 'empty struct array');
for i = 1:5
    news = struct('v', {10*i, 20*i});
    T = [T; news];
end
assert(numel(T) == 10, sprintf('grown numel = %d', numel(T)));
% Values should include 10, 20, 20, 40, 30, 60, ... but we don't enforce
% order since numbl's flat struct array differs from MATLAB's 2D shape.
allv = [T.v];
assert(sum(allv) == 450, sprintf('sum = %g', sum(allv)));

% --- Mismatched field names should error ---
threw = false;
try
    bad = [s, struct('a', 1, 'c', 2)];  %#ok<NASGU>
catch
    threw = true;
end
assert(threw, 'mismatched fields should error');

disp('SUCCESS');
