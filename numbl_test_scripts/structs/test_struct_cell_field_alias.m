% Regression: `struct(field1, e, field2, e)` where `e` is a cell creates
% a struct array whose elements share field references — both fields
% point to the same cell entry. If the cell entry is a tensor, recursive
% dispose at function exit (or via `clear all`) double-disposes that
% tensor.
%
% Reproduces the bug seen in flam/hypoct.m line 101:
%   s = struct('ctr',e,'xi',e,'prnt',e,'chld',e,'nbor',e);
% where `e = cell(M,1)` — each struct array element ends up with all
% five fields aliased to the same value.

e = cell(2, 1);
e{1} = [1 2];
e{2} = [3 4];

% struct() with the same cell `e` for both fields used to alias
% s(k).a and s(k).b to the SAME tensor reference. Disposing s would
% then dispose each shared tensor twice.
s = struct('a', e, 'b', e);

% Sanity check the value first.
assert(isequal(s(1).a, [1 2]));
assert(isequal(s(1).b, [1 2]));
assert(isequal(s(2).a, [3 4]));
assert(isequal(s(2).b, [3 4]));

% Mutating one field on one element must not corrupt the other field
% (independence test — would fail if fields are aliased).
s(1).a = [99 99];
assert(isequal(s(1).b, [1 2]), 's(1).b should still be [1 2] after mutating s(1).a');

% This `clear all` triggers recursive dispose. With the bug, the shared
% tensor refs cause DoubleDisposeError on dispose; with the fix, every
% field is independent so dispose runs cleanly.
clear all;

disp('SUCCESS');
