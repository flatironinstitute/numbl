% Test tensor power operations in JIT

function y = elem_pow(a, b)
    y = a .^ b;
end

%!jit
r = elem_pow([1 2 3 4], 2);
assert(all(r == [1 4 9 16]));

%!jit
r2 = elem_pow([1 4 9], 0.5);
assert(all(abs(r2 - [1 2 3]) < 1e-10));

%!jit
% tensor .^ tensor
function y = elem_pow_tt(a, b)
    y = a .^ b;
end
r3 = elem_pow_tt([2 3 4], [1 2 3]);
assert(all(r3 == [2 9 64]));

%!jit
% scalar ^ tensor (uses Pow, not ElemPow, but scalar*tensor is element-wise)
function y = pow_st(s, t)
    y = s ^ t;
end
r4 = pow_st(2, [1 2 3]);
assert(all(r4 == [2 4 8]));

disp('SUCCESS');
