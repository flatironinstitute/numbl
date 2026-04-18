% C-JIT parity gap #03: AssignIndex on a pure-input tensor param
% (the param is written to but is NOT also a named output).
%
% The JS-JIT compiles this by unshare-at-entry (copy-on-write promotes
% the param's tensor to a writable local copy); the C-JIT currently
% bails feasibility with
%   "AssignIndex base 'v' must be both param and output"
% because the existing parity02 wrapper only seeds the output buffer
% when output name == param name.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 99
%   numbl --opt 2 run <this>                         -> 99  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> ERRORS (the gap)
%   matlab -batch parity03_input_param_assign_index  -> 99
%
% Caller-side v must stay unchanged (MATLAB call-by-value); we also
% assert that after the call by calling mutate_scalar three times with
% the same caller-side v.

v = [1, 2, 3, 4, 5];
y = mutate_scalar(v);
y = mutate_scalar(v);
y = mutate_scalar(v);
assert(isequal(v, [1, 2, 3, 4, 5]), 'caller v must be unchanged');
assert(isequal(y, [1, 2, 99, 4, 5]), 'returned y must carry the mutation');
disp(y(3))

function out = mutate_scalar(v)
    v(3) = 99;
    out = v;
end
