% C-JIT parity gap #12: UserCall with scalar args + scalar return.
%
% The C-JIT historically bailed feasibility with
%   "unsupported expr: UserCall"
% Any call to a user-defined function from a C-JIT-feasible outer
% forced the whole outer back to JS-JIT. The UserCall iteration
% emits each reachable callee as a `static void jit_<jitName>(...)`
% in the same .c file, linked directly — no JS callback, no koffi
% boundary.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 35\n28\n10
%   numbl --opt 2 run <this>                         -> 35\n28\n10
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 35\n28\n10
%   matlab -batch parity12_user_call_scalar          -> 35\n28\n10

% 1) Scalar UserCall inside a for-loop: the classic trigger.
total = 0;
for i = 1:5
    total = total + inner_sum(i);
end
disp(total)      % 35

% 2) UserCall with multiple scalar args.
disp(clamp(42, 0, 28))   % 28

% 3) Nested UserCall (callee A calls callee B). Post-order emission
%    must produce B before A so the C compiler sees B when A refs it.
disp(double_sum(4))      % twice(5) = 10

function r = inner_sum(n)
    r = 0;
    for k = 1:n
        r = r + k;
    end
end

function r = clamp(x, lo, hi)
    if x < lo
        r = lo;
    elseif x > hi
        r = hi;
    else
        r = x;
    end
end

function r = double_sum(n)
    r = twice(n + 1);
end

function r = twice(n)
    r = n * 2;
end
