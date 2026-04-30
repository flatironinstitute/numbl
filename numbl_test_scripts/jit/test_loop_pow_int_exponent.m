% Integer-exponent power (`.^ 2` etc.) on a real tensor of unknown sign.
%
% Before fix: binaryResultType for ElemPow bailed when the base couldn't
% be proved nonneg, even though Math.pow with an integer exponent is
% always real (no NaN risk). The chunkie hot path uses `dint .^ 2` to
% compute squared lengths and was bailing here.

function jit_pow_int_main()
    a = randn(2, 16);  % can be negative
    total = 0;
    for ii = 1:5
        %!numbl:assert_jit
        sq = a .^ 2;
        cu = a .^ 3;
        total = total + sum(sq(:)) + sum(cu(:));
    end
    expected = sum(a(:) .^ 2) + sum(a(:) .^ 3);
    assert(abs(5 * expected - total) < 1e-9, '1: integer exponent JIT result');
end

jit_pow_int_main();
disp('SUCCESS');
