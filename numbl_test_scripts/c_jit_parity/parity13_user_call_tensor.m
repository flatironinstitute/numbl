% C-JIT parity gap #13: UserCall with tensor args and/or tensor return.
%
% Scalar UserCall parity closed in parity12. This iteration widens the
% call-site emitter to (a) pass tensor Vars to callees via data + len
% (+ d0/d1 when the callee reads as a matrix) and (b) accept
% fresh-alloc tensor returns via the dynamic-output ABI. Covers the
% four shapes that matter for real-world scientific code:
%   1) tensor-return (pure scalar args, tensor result).
%   2) tensor-arg + tensor-return.
%   3) tensor-return inside a hot loop (the classic speedup target).
%   4) nested: A calls B, both take+return tensors.
%
% Expected disp output (all runs):
%   numbl --opt 1 run <this>                         -> 15\n30\n10\n30
%   numbl --opt 2 run <this>                         -> 15\n30\n10\n30
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 15\n30\n10\n30
%   matlab -batch parity13_user_call_tensor          -> 15\n30\n10\n30

% 1) Tensor-return: callee mallocs, transfers ownership via `double**`.
v = make_range(5);
disp(sum(v))     % 1+2+3+4+5 = 15

% 2) Tensor-arg + tensor-return round-trip.
w = scale_by(v, 2);
disp(sum(w))     % 2+4+6+8+10 = 30

% 3) Inside a for-loop: the hot path. Each iteration gets its own fresh
%    buffer; the epilogue free() machinery reclaims the previous one.
total = 0;
for i = 1:3
    x = make_range(i);
    total = total + sum(x);
end
disp(total)      % 1 + 3 + 6 = 10

% 4) Nested: make_scaled internally calls both make_range and scale_by.
z = make_scaled(4);
disp(sum(z))     % scale_by(make_range(4), 3) = 3+6+9+12 = 30

function r = make_range(n)
    r = zeros(1, n);
    for k = 1:n
        r(k) = k;
    end
end

function w = scale_by(v, s)
    w = zeros(1, length(v));
    for k = 1:length(v)
        w(k) = v(k) * s;
    end
end

function z = make_scaled(n)
    v = make_range(n);
    z = scale_by(v, 3);
end
