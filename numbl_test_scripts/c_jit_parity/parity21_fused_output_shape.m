% C-JIT parity gap #21: `--fuse` dropped dynamic-output shape (d0/d1).
%
% Under `--fuse`, the fused per-element codegen emitted a dynamic-output
% tensor but never wrote `v_y_d0` / `v_y_d1`. The JS wrapper read those
% zero-initialized locals back into the RuntimeTensor shape, so
% `size(y)` returned `[0 0]` while `numel(y)` returned the correct N.
%
% Expected disp output (must match across all runs):
%   numbl --opt 1 run <this>            -> 1\n3\n1\n5\nSUCCESS
%   numbl --opt 2 --fuse run <this>     -> 1\n3\n1\n5\nSUCCESS
%   matlab -batch parity21_fused_output_shape -> 1\n3\n1\n5\nSUCCESS

% 1) Row-vector source, real: size must be [1 3] (not [0 0]).
x = [1 2 3];
for k = 1:4
    y = chain(x);
end
disp(size(y, 1))
disp(size(y, 2))
assert(size(y, 1) == 1, 'row d0');
assert(size(y, 2) == 3, 'row d1');
assert(numel(y) == 3, 'row numel');

% 2) Column-vector source: size must be [5 1].
x2 = (1:5)';
for k = 1:4
    y2 = chain(x2);
end
disp(size(y2, 1))
disp(size(y2, 2))
assert(size(y2, 1) == 5, 'col d0');
assert(size(y2, 2) == 1, 'col d1');

disp('SUCCESS')

function y = chain(x)
    y = x .* x + 2;
    y = y .* y + 3;
end
