% TEST: element-wise chains that the C-JIT inline pass folds into one
% fused loop (the kernel pass, lowering/elemwiseKernels.ts). Covers: a loop-body def that is
% also read after the loop (must not be dropped), a consumer that
% overwrites an operand of the folded temps, element-wise builtins in the
% chain, and dynamic-extent operands whose runtime shapes differ (the
% fused loop's shape guard falls back to the broadcasting helpers, now
% with a nested RHS). Before wiring, the pass counted uses per body, so it
% would have dropped the loop-body def of `w` and `q` would diverge at opt2.
a = [0.1 0.2 0.3 0.4 0.5];
b = [1.5 -2.25 3.125 0.7 -0.3];
z = a;
v = a;
for k = 1:3
    w = a .* b + k;
    z = w .* 2 - z;
    v = v + 0.01 .* (a .* v - b);
end
q = w + 1;
u = exp(-a .* a) .* cos(b) + max(a - 0.3, 0);
for n = 1:4:5
    p = ones(1, n) .* 3;
    s = linspace(1, 2, 5 - 0 * n);
    r = (s .* s + p) .* p - s;
    fprintf('%.17g ', r); fprintf('\n');
end
fprintf('%.17g ', z); fprintf('\n');
fprintf('%.17g ', v); fprintf('\n');
fprintf('%.17g ', q); fprintf('\n');
fprintf('%.17g ', u); fprintf('\n');
