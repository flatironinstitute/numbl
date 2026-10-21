% discretize: bins [e_k, e_{k+1}), last bin closed, NaN outside
e = [0 1 2 3];
x = [-1 0 0.5 1 2.5 3 3.5 NaN];
y = discretize(x, e);
assert(eqn(y, [NaN 1 1 2 3 3 NaN NaN]));
assert(isa(y, 'double'));

% shape follows X
assert(eqn(discretize(x', e), [NaN 1 1 2 3 3 NaN NaN]'));
assert(isequal(discretize([1 1; 2 0], e), [2 2; 3 1]));
assert(discretize(2.5, e) == 3);
assert(isequal(size(discretize(zeros(0, 3), e)), [0 3]));

% IncludedEdge right: (e_k, e_{k+1}], first bin closed
assert(eqn(discretize(x, e, 'IncludedEdge', 'right'), [NaN 1 1 1 3 3 NaN NaN]));

% values
assert(eqn(discretize(x, e, [10 20 30]), [NaN 10 10 20 30 30 NaN NaN]));

% repeated edges
assert(discretize(1, [0 1 1]) == 2);
assert(isequal(discretize([0 1 2], [0 1 1 2]), [1 3 3]));
assert(isequal(discretize([0 1], [0 0 1], 'IncludedEdge', 'right'), [1 2]));

% second output: the edges
[y, E] = discretize(2, e');
assert(y == 3);
assert(isequal(E, e'));

disp('SUCCESS');

function tf = eqn(a, b)
% isequal treating NaNs as equal
tf = isequal(size(a), size(b)) && isequal(isnan(a), isnan(b)) && ...
    isequal(a(~isnan(a)), b(~isnan(b)));
end
