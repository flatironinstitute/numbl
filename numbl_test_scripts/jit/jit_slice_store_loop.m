% `A(k, :) = rhs` with a loop-variable row index compiles.
%
% The lowerer used to require a *statically provable* in-bounds index for
% every scalar slot of a slice store, and a loop variable never is — so the
% commonest MATLAB row-filling idiom kept its whole loop (and everything
% nested inside it) on the interpreter. Both backends already route a
% store's scalar slot through the grow-aware bounds check, so the guarantee
% is kept at runtime instead: in bounds it writes, past the extent it bails
% to the interpreter (which grows), below 1 it errors.

A = zeros(4, 3);
for k = 1:4
    %!numbl:assert_jit
    A(k, :) = [k, 2*k, 3*k];
end
assert(sum(A(:)) == 60, 'row store');
assert(A(3, 2) == 6, 'row store value');

B = zeros(3, 4);
for k = 1:4
    %!numbl:assert_jit
    B(:, k) = k;
end
assert(sum(B(:)) == 30, 'column store, scalar broadcast');

% A write past the extent still grows the array — the JIT bails and the
% interpreter takes over, so the result matches --opt 0.
C = zeros(2, 2);
for k = 1:4
    C(k, :) = [k, k];
end
assert(isequal(size(C), [4, 2]), 'grow bail keeps MATLAB semantics');
assert(C(4, 1) == 4, 'grown row value');

% Sub-1 index is a genuine error, not a grow.
threw = false;
try
    D = zeros(3, 3);
    j = 0;
    D(j, :) = [1 2 3];
catch
    threw = true;
end
assert(threw, 'index 0 must error');

disp('SUCCESS')
