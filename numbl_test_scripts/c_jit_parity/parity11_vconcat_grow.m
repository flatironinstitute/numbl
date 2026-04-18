% C-JIT parity gap #11: VConcatGrow (`[base; value]`).
%
% The JS-JIT lowers the chunkie "grow-a-list" pattern
%     it = [];
%     for ...
%         it = [it; i];
%     end
% to a `VConcatGrow` IR node emitted as `$h.vconcatGrow1r(base, value)`.
% The C-JIT historically bailed feasibility with
%   "unsupported expr: VConcatGrow"
% because the inner Assign reallocates it's buffer each iteration and
% there was no self-grow codegen. This gap closes via the dynamic-
% output ABI plus inline malloc/memcpy/append codegen (guarded so the
% memcpy from base completes before the old buffer is freed, which
% makes self-grow safe).
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 20\n1\n20\n30\n165
%   numbl --opt 2 run <this>                         -> 20\n1\n20\n30\n165
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 20\n1\n20\n30\n165
%   matlab -batch parity11_vconcat_grow              -> 20\n1\n20\n30\n165

% 1) Basic grow: column vector 1..20 via `[it; i]`.
it = [];
for i = 1:20
    it = [it; i];
end
disp(length(it))         % 20
disp(it(1))              % 1
disp(it(20))             % 20

% 2) Conditional growth plus sum — the classic chunkie found-list shape.
it2 = [];
for i = 1:50
    if mod(i, 5) == 0
        it2 = [it2; i];
    end
end
assert(length(it2) == 10, 'length of mod-5 hits in 1..50');
% Sum via explicit accumulator to verify per-element reads line up.
acc = 0;
for k = 1:length(it2)
    acc = acc + it2(k);
end
assert(acc == 5 + 10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50, 'it2 sum');
disp(length(it2) * 3)    % 30 (just a shape check)

% 3) Reset + grow across outer iters: inner it3 is reset to `[]` and
%    rebuilt each outer iter. Mirrors parity01 of chunkie's per-leaf
%    reset, with a scalar cross-iter accumulator for the result.
acc = 0;
for i = 1:10
    it3 = [];
    it3 = [it3; i];
    it3 = [it3; i * 2];
    acc = acc + it3(1) + it3(2);
end
% it3(1) = i, it3(2) = 2i, so acc = sum(i + 2i for i=1..10) = 3 * 55 = 165.
disp(acc)                % 165

disp('SUCCESS');
