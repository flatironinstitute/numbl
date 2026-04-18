% C-JIT parity gap #01: scalar tensor Index read inside a user function.
%
% The JS-JIT compiles `y = x(i)` via runtimeIndexing fast paths, but the
% C-JIT feasibility checker bails with
%   "Index reads not supported (defer to JS-JIT)"
% in src/numbl-core/jit/c/cFeasibility.ts.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 30
%   numbl --opt 2 run <this>                         -> 30  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> ERRORS (the gap)
%   matlab -batch parity01_index_read                -> 30
%
% Once the C-JIT emits Index reads, the --check-c-jit-parity run will
% pass and print 30, closing this gap.

x = [10, 20, 30, 40, 50];
val = get_element(x, 3);
disp(val)

function y = get_element(x, i)
    y = x(i);
end
