% Indexed assignment and element/row/column deletion must preserve a logical
% array's class (MATLAB casts the RHS to logical and keeps the array logical).
% This held for uniquely-owned arrays but was lost when the array was shared
% (copy-on-write) or after deletion -- the rktoolbox rat_krylov failure, where
% a logical `column_deflation` mask silently became a double index vector.

% --- element deletion keeps logical ---
L = false(1, 8);
L(1) = true;
L(7:end) = [];
assert(islogical(L), 'element deletion must keep logical');
assert(numel(L) == 6 && L(1) == true, 'element deletion values');

% --- row / column deletion keeps logical ---
M = logical([1 0 1; 0 1 0]);
M(1, :) = [];
assert(islogical(M), 'row deletion must keep logical');
P = logical([1 0 1; 0 1 0]);
P(:, 2) = [];
assert(islogical(P), 'column deletion must keep logical');

% --- indexed assignment into a SHARED logical array keeps logical ---
% Aliasing forces copy-on-write on the first write.
base = false(1, 6);
alias = base;          %#ok<NASGU>  forces base to be shared
base(2) = 1;           % double RHS, must cast to logical, stay logical
assert(islogical(base), 'shared indexed-assign must keep logical');
assert(base(2) == true, 'value cast to logical true');

% --- the rktoolbox pattern: logical mask through a struct field that is
%     shared across a function boundary, then used as a column index ---
p = make_param();      % returns a struct holding a logical mask
p.mask(1) = 1;         % indexed write on the shared field
assert(islogical(p.mask), 'struct-field shared logical kept');
K = reshape(1:42, 7, 6);
sub = K(1:2, p.mask);  % logical column indexing selects column 1
assert(isequal(sub, [1; 2]), 'logical column index after shared write');

disp('SUCCESS');

function p = make_param()
    p.mask = false(1, 6);
end
