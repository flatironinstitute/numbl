% C-JIT parity gap #19: variable-dim shape sentinel leaks as literal -1.
%
% When a JIT'd function is specialized on an input whose shape has an
% unknown (sentinel = -1) dimension, `shapeExprsFor` in emit.ts stringified
% the -1 into C code, so the dynamic-output d0/d1 carried the literal
% -1 back to JS. `size(y)` then reported negative values.
%
% Expected disp output (must match across all runs):
%   numbl --opt 1 run <this>  -> 3\n4\n5\n6\n7\nSUCCESS
%   numbl --opt 2 run <this>  -> 3\n4\n5\n6\n7\nSUCCESS
%   matlab -batch parity19_variable_shape_passthrough -> 3\n4\n5\n6\n7\nSUCCESS
%
% Bug symptom (before fix, --opt 2): 2nd and later calls reported
%   size(y) = [-1 1]
% because `hot_chain` got recompiled with shape [?x1] and the unknown
% d0 was emitted into C as `v_y_d0 = -1;`.

for n = 3:7
    y = doubler(ones(n, 1));
    assert(size(y, 1) == n, 'd0 should equal input d0');
    assert(size(y, 2) == 1, 'd1 should be 1');
    disp(size(y, 1))
end

disp('SUCCESS')

function y = doubler(x)
    y = 2 * x;
end
