% C-JIT parity gap #02: scalar tensor Index write inside a user function.
%
% The JS-JIT compiles `x(i) = v` via set1r_h (soft-bails on OOB so the
% interpreter can grow the tensor); the C-JIT currently bails feasibility
% with "unsupported stmt: AssignIndex".
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 99
%   numbl --opt 2 run <this>                         -> 99  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> ERRORS (the gap)
%   matlab -batch parity02_index_write               -> 99
%
% Once the C-JIT emits AssignIndex, the --check-c-jit-parity run will
% pass and print 99, closing this gap.

x = [10, 20, 30, 40, 50];
y = set_element(x, 3, 99);
disp(y(3))

function x = set_element(x, i, v)
    x(i) = v;
end
