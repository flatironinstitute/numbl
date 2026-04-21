% C-JIT parity gap #22: int64 overflow in zeros(n, m) silently corrupts
% the output (numel wraps while d0/d1 remain huge).
%
% `zeros(n, m)` in emit.ts computes `__zn = __zr * __zc` as int64 with
% no overflow detection. For n = m = 2^32, __zn wraps to 0; the code
% then calls calloc(0, 8) and reports size = [2^32 2^32] with numel = 0.
% The interpreter and MATLAB both error on this input; the C-JIT must
% too (err_flag = 1.0 → hard bounds error).
%
% Expected disp output (must match across all runs):
%   numbl --opt 1 run <this>  -> SUCCESS
%   numbl --opt 2 run <this>  -> SUCCESS
%   matlab -batch parity22_zeros_int_overflow -> SUCCESS

% Warm the JIT with small sizes.
for k = 1:5
    z = mkz(2, 3);
    assert(isequal(size(z), [2 3]));
end

% Overflow: 2^32 * 2^32 wraps to 0 as int64. Must throw.
threw = false;
try
    z = mkz(4294967296, 4294967296); %#ok<NASGU>
catch
    threw = true;
end
assert(threw, 'zeros overflow must throw');

disp('SUCCESS')

function z = mkz(n, m)
    z = zeros(n, m);
end
