% Test tensor comparison operators in JIT

function y = compare_ops(a, b)
    eq = a == b;
    ne = a ~= b;
    lt = a < b;
    le = a <= b;
    gt = a > b;
    ge = a >= b;
    y = eq + ne + lt + le + gt + ge;
end

%!jit
x = [1 2 3 4];
v = [2 2 2 2];
r = compare_ops(x, v);
% eq: [0 1 0 0], ne: [1 0 1 1], lt: [1 0 0 0], le: [1 1 0 0], gt: [0 0 1 1], ge: [0 1 1 1]
assert(all(r == [3 3 3 3]));

%!jit
% scalar vs tensor
function y = scalar_cmp(a, s)
    y = a > s;
end
r2 = scalar_cmp([1 5 3 7], 4);
assert(all(r2 == [0 1 0 1]));

%!jit
% tensor vs tensor same shape
function y = mat_cmp(a, b)
    y = a <= b;
end
r3 = mat_cmp([1 2; 3 4], [2 1; 3 5]);
assert(all(all(r3 == [1 0; 1 1])));

disp('SUCCESS');
