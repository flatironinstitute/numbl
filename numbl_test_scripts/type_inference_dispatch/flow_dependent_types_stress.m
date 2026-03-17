% Stress tests for flow-dependent type inference.
% Targets edge cases and patterns likely to break the fixpoint/widening logic.

% =====================================================================
% 1. Loop variable swap pattern (Legendre recurrence pattern)
%    a starts Number, b starts Tensor. After swap, a becomes Tensor.
%    Both should widen to Unknown.
% =====================================================================

a_swap = 1;
b_swap = [1 2 3];
for k = 1:2
    tmp = a_swap;
    a_swap = b_swap;
    b_swap = tmp;
end
assert(strcmp(__inferred_type_str(a_swap), 'Unknown'));
assert(strcmp(__inferred_type_str(b_swap), 'Unknown'));

% =====================================================================
% 2. Loop that executes zero times (empty range)
%    Types should stay at pre-loop values.
% =====================================================================

zr = 42;
for k = 1:0
    zr = 'changed';
end
% Loop body never executes, but static analysis sees possible type change
assert(strcmp(__inferred_type_str(zr), 'Unknown'));
assert(zr == 42);  % runtime value unchanged

% =====================================================================
% 3. While loop that executes zero times
% =====================================================================

wz = 'hello';
while false
    wz = 99;
end
assert(strcmp(__inferred_type_str(wz), 'Unknown'));
assert(strcmp(wz, 'hello'));  % runtime value unchanged

% =====================================================================
% 4. Multiple assignments to same variable in one branch
%    Last assignment should determine the branch's type contribution.
% =====================================================================

multi = 0;
if true
    multi = 'temp';
    multi = [1 2 3];
    multi = 42;
else
    multi = 99;
end
% Then-branch ends with Number (42), else-branch is Number (99)
% Pre-branch is Number (0). All Number -> Number
assert(strcmp(__inferred_type_str(multi), 'Number'));

% =====================================================================
% 5. Reassignment in loop body that stays same type
%    despite intermediate different type
% =====================================================================

stable_loop = 0;
for k = 1:3
    stable_loop = 'temp';
    stable_loop = k + 1;  % back to Number
end
% Body ends with Number (k+1), pre-loop is Number.
% Post-loop join: Number + Number = Number.
% The intermediate Char assignment only affects mid-body type.
assert(strcmp(__inferred_type_str(stable_loop), 'Number'));

% =====================================================================
% 6. Nested if inside loop - type change only in inner branch
% =====================================================================

nl = 0;
for k = 1:3
    if k > 10  % never true at runtime, but statically unknown
        nl = 'changed';
    end
end
% nl could be 'changed' (Char) in if-branch, or stay Number (from pre-if)
% if without else -> join pre-branch(Number) with then(Char) -> Unknown in loop
% Then loop join: pre-loop(Number) vs body(Unknown) -> Unknown
assert(strcmp(__inferred_type_str(nl), 'Unknown'));

% =====================================================================
% 7. Consecutive control flow blocks
% =====================================================================

seq = 0;
if true
    seq = 1;
end
% After first if: Number (same type both paths)
assert(strcmp(__inferred_type_str(seq), 'Number'));

if true
    seq = 'now_char';
else
    seq = 'also_char';
end
% After second if: Char (same type both branches)
assert(strcmp(__inferred_type_str(seq), 'Char'));

seq = 100;
% After reassignment: Number again
assert(strcmp(__inferred_type_str(seq), 'Number'));

% =====================================================================
% 8. Variable only assigned in else branch (not then)
% =====================================================================

eo = 'initial';
if true
    % no assignment to eo
else
    eo = 42;
end
% Pre-branch is Char, then-branch doesn't assign (stays Char from pre),
% else-branch is Number. With else present, join is then vs else only?
% Actually: if/else with else present -> join then-post and else-post
% then-post: eo = Char (unchanged), else-post: eo = Number -> Unknown
assert(strcmp(__inferred_type_str(eo), 'Unknown'));

% =====================================================================
% 9. Chain of type changes through multiple loops
% =====================================================================

chain = 0;
for k = 1:1
    chain = chain + 1;
end
assert(strcmp(__inferred_type_str(chain), 'Number'));
assert(chain == 1);

chain = 'reset';
assert(strcmp(__inferred_type_str(chain), 'Char'));

for k = 1:1
    chain = 42;
end
% Pre-loop Char, body Number -> Unknown
assert(strcmp(__inferred_type_str(chain), 'Unknown'));

% =====================================================================
% 10. Deeply nested: for > if > for > if
% =====================================================================

dd = 0;
for i = 1:2
    if true
        for j = 1:2
            if true
                dd = dd + 1;
            end
        end
    end
end
% dd + 1 is always Number + Number = Number, no type change
assert(strcmp(__inferred_type_str(dd), 'Number'));
assert(dd == 4);

% =====================================================================
% 11. Nested loops with cross-variable dependencies
% =====================================================================

outer_v = 0;
inner_v = 0;
for i = 1:3
    outer_v = outer_v + inner_v;
    for j = 1:3
        inner_v = inner_v + 1;
    end
end
% Both stay Number throughout
assert(strcmp(__inferred_type_str(outer_v), 'Number'));
assert(strcmp(__inferred_type_str(inner_v), 'Number'));

% =====================================================================
% 12. While loop with break
% =====================================================================

wb = 0;
wi2 = 0;
while true
    wb = wb + 1;
    wi2 = wi2 + 1;
    if wi2 >= 3
        break;
    end
end
assert(strcmp(__inferred_type_str(wb), 'Number'));
assert(wb == 3);

% =====================================================================
% 13. For loop where body assigns iteration variable
% =====================================================================

for kk = 1:5
    kk2 = kk * 2;
end
assert(strcmp(__inferred_type_str(kk2), 'Number'));

% =====================================================================
% 14. Switch inside for loop
% =====================================================================

sw_in_loop = 0;
for k = 1:3
    switch k
        case 1
            sw_in_loop = sw_in_loop + 10;
        case 2
            sw_in_loop = sw_in_loop + 20;
        otherwise
            sw_in_loop = sw_in_loop + 30;
    end
end
assert(strcmp(__inferred_type_str(sw_in_loop), 'Number'));
assert(sw_in_loop == 60);

% =====================================================================
% 15. For loop inside switch
% =====================================================================

loop_in_sw = 0;
switch 2
    case 1
        for k = 1:5
            loop_in_sw = loop_in_sw + 1;
        end
    case 2
        for k = 1:10
            loop_in_sw = loop_in_sw + 1;
        end
    otherwise
        loop_in_sw = 99;
end
assert(strcmp(__inferred_type_str(loop_in_sw), 'Number'));
assert(loop_in_sw == 10);

% =====================================================================
% 16. Try/catch inside loop
% =====================================================================

tc_loop = 0;
for k = 1:3
    try
        tc_loop = tc_loop + k;
    catch
        tc_loop = -1;
    end
end
assert(strcmp(__inferred_type_str(tc_loop), 'Number'));
assert(tc_loop == 6);

% =====================================================================
% 17. Loop inside try/catch
% =====================================================================

loop_tc = 0;
try
    for k = 1:5
        loop_tc = loop_tc + k;
    end
catch
    loop_tc = -1;
end
assert(strcmp(__inferred_type_str(loop_tc), 'Number'));
assert(loop_tc == 15);

% =====================================================================
% 18. Accumulator pattern: tensor grows via concatenation
% =====================================================================

acc = [];
for k = 1:3
    acc = [acc, k];
end
assert(strcmp(__inferred_type_str(acc), 'Tensor<?, real>'));
assert(numel(acc) == 3);

% =====================================================================
% 19. Numeric accumulation after type-changing control flow
%     Variable starts Unknown, then we reassign to Number
% =====================================================================

q = 0;
if true
    q = 'temp';
else
    q = 42;
end
assert(strcmp(__inferred_type_str(q), 'Unknown'));

q = 0;
assert(strcmp(__inferred_type_str(q), 'Number'));
for k = 1:5
    q = q + k;
end
assert(strcmp(__inferred_type_str(q), 'Number'));
assert(q == 15);

% =====================================================================
% 20. Struct field assignment inside control flow
% =====================================================================

st = struct();
st.x = 1;
if true
    st.y = 'hello';
else
    st.y = 'world';
end
assert(strcmp(__inferred_type_str(st), 'Struct<x: Number, y: Char>'));

% =====================================================================
% 21. Three-variable rotation in loop (stress fixpoint)
%     a->b->c->a pattern that needs multiple fixpoint iterations
% =====================================================================

rot_a = 1;          % Number
rot_b = [1 2];      % Tensor
rot_c = 'abc';      % Char
for k = 1:2
    rot_tmp = rot_a;
    rot_a = rot_b;
    rot_b = rot_c;
    rot_c = rot_tmp;
end
% All three should be Unknown (each can hold any of the three types)
assert(strcmp(__inferred_type_str(rot_a), 'Unknown'));
assert(strcmp(__inferred_type_str(rot_b), 'Unknown'));
assert(strcmp(__inferred_type_str(rot_c), 'Unknown'));
assert(strcmp(__inferred_type_str(rot_tmp), 'Unknown'));

% =====================================================================
% 22. While loop with complex condition involving loop variable
% =====================================================================

wc = 0;
while wc < 10
    wc = wc + 3;
end
assert(strcmp(__inferred_type_str(wc), 'Number'));
assert(wc == 12);

% =====================================================================
% 23. Elseif where only some branches assign
% =====================================================================

ei = 100;
if false
    ei = 200;
elseif false
    % no assignment to ei
elseif true
    ei = 300;
end
% Pre-branch: Number. Branches: 200(Number), unchanged(Number), 300(Number)
% No else -> includePreBranch. All Number -> Number
assert(strcmp(__inferred_type_str(ei), 'Number'));
assert(ei == 300);

% =====================================================================
% 24. Elseif where middle branch changes type
% =====================================================================

ei2 = 0;
if false
    ei2 = 1;
elseif false
    ei2 = 'changed';
else
    ei2 = 3;
end
% Branches: Number, Char, Number. Has else -> join branches only -> Unknown
assert(strcmp(__inferred_type_str(ei2), 'Unknown'));

% =====================================================================
% 25. Nested while loops (both same variable)
% =====================================================================

nw = 0;
oi = 0;
while oi < 2
    oi = oi + 1;
    ij = 0;
    while ij < 3
        ij = ij + 1;
        nw = nw + 1;
    end
end
assert(strcmp(__inferred_type_str(nw), 'Number'));
assert(nw == 6);

% =====================================================================
% 26. Variable first assigned inside loop (not pre-defined)
% =====================================================================

for k = 1:3
    first_in_loop = k;
end
assert(strcmp(__inferred_type_str(first_in_loop), 'Number'));
assert(first_in_loop == 3);

% =====================================================================
% 27. Conditional that narrows back to same type
% =====================================================================

narrow = [1 2 3];
if true
    narrow = [4 5 6];
else
    narrow = [7 8 9];
end
assert(strcmp(__inferred_type_str(narrow), 'Tensor<?, real>'));

% =====================================================================
% 28. Expression involving variable whose type was widened
% =====================================================================

ew = 10;
if true
    ew = 'str';
else
    ew = 20;
end
% ew is Unknown here
assert(strcmp(__inferred_type_str(ew), 'Unknown'));

% =====================================================================
% 29. Mixed tensor operations after precise type preservation
% =====================================================================

mat = [1 2; 3 4];
assert(strcmp(__inferred_type_str(mat), 'Tensor<?, real>'));
mat2 = mat * 2;
assert(strcmp(__inferred_type_str(mat2), 'Tensor<?, real>'));
mat3 = mat + mat2;
assert(strcmp(__inferred_type_str(mat3), 'Tensor<?, real>'));

% =====================================================================
% 30. Runtime correctness: loop body arithmetic must work even when
%     types were widened to Unknown at compile time
% =====================================================================

x_rt = 1;
y_rt = [1 2 3 4 5];
for k = 1:3
    tmp_rt = x_rt;
    x_rt = y_rt;
    y_rt = tmp_rt;
end
% After 3 iterations: x_rt=tensor, y_rt=scalar (odd number of swaps)
assert(numel(x_rt) == 5);
assert(y_rt == 1);

% =====================================================================
% 31. Multiple control flow blocks in sequence, each with different vars
% =====================================================================

s1v = 0;
s2v = 0;
s3v = 0;
if true
    s1v = 'a';
end
for k = 1:1
    s2v = [1 2];
end
while false
    s3v = true;
end
assert(strcmp(__inferred_type_str(s1v), 'Unknown'));
assert(strcmp(__inferred_type_str(s2v), 'Unknown'));
assert(strcmp(__inferred_type_str(s3v), 'Unknown'));

% =====================================================================
% 32. Boolean variable through control flow
% =====================================================================

flag = true;
if true
    flag = false;
end
assert(strcmp(__inferred_type_str(flag), 'Boolean'));

flag2 = true;
if true
    flag2 = 0;
end
assert(strcmp(__inferred_type_str(flag2), 'Unknown'));

% =====================================================================
% 33. Verify runtime correctness of Legendre-style recurrence
%     (the actual pattern that broke chebfun)
% =====================================================================

x_leg = linspace(0, 1, 5);
Pm2_leg = 1;            % scalar
Pm1_leg = x_leg;        % tensor

for k = 1:3
    P_leg = ((2*k+1)*Pm1_leg.*x_leg - k*Pm2_leg) / (k+1);
    Pm2_leg = Pm1_leg;
    Pm1_leg = P_leg;
end

% P_leg should be P_4(x) = (35x^4 - 30x^2 + 3)/8
P4_exact = (35*x_leg.^4 - 30*x_leg.^2 + 3) / 8;
assert(max(abs(P_leg - P4_exact)) < 1e-10, 'Legendre P4 mismatch');
assert(numel(P_leg) == 5);

% =====================================================================
% 34. While+for nested (Newton iteration pattern)
% =====================================================================

x_nw = [0.8; 0.3];
Pm2_nw = 1;
Pm1_nw = x_nw;
n_nw = 4;
dx_nw = inf;
ctr = 0;
PPm2_nw = 0;
PPm1_nw = 1;

while norm(dx_nw, inf) > eps && ctr < 10
    ctr = ctr + 1;
    for k = 1:n_nw-1
        P_nw = ((2*k+1)*Pm1_nw.*x_nw - k*Pm2_nw) / (k+1);
        Pm2_nw = Pm1_nw;
        Pm1_nw = P_nw;
        PP_nw = ((2*k+1)*(Pm2_nw + x_nw.*PPm1_nw) - k*PPm2_nw) / (k+1);
        PPm2_nw = PPm1_nw;
        PPm1_nw = PP_nw;
    end
    dx_nw = -P_nw./PP_nw;
    x_nw = x_nw + dx_nw;
    Pm2_nw = 1;
    Pm1_nw = x_nw;
    PPm2_nw = 0;
    PPm1_nw = 1;
end

assert(~any(isnan(x_nw)), 'Newton should not produce NaN');
assert(ctr > 0 && ctr <= 10, 'Newton should converge');
P4_roots = (35*x_nw.^4 - 30*x_nw.^2 + 3) / 8;
assert(max(abs(P4_roots)) < 1e-10, 'Should converge to Legendre roots');

% =====================================================================
% 35. For loop with conditional break (tests type after early exit)
% =====================================================================

fb = 0;
for k = 1:100
    fb = fb + 1;
    if fb >= 5
        break;
    end
end
assert(strcmp(__inferred_type_str(fb), 'Number'));
assert(fb == 5);

% =====================================================================
% 36. Double-nested if with type changes at different levels
% =====================================================================

dn = 0;
if true
    if true
        dn = 1;
    else
        dn = 'inner';
    end
    % After inner if: Unknown (Number vs Char)
else
    dn = 2;
end
% After outer if: join(Unknown from then, Number from else) = Unknown
assert(strcmp(__inferred_type_str(dn), 'Unknown'));

% Same structure but all numbers
dn2 = 0;
if true
    if true
        dn2 = 1;
    else
        dn2 = 2;
    end
else
    dn2 = 3;
end
assert(strcmp(__inferred_type_str(dn2), 'Number'));

% =====================================================================
% 37. Variable used in loop condition that gets widened
% =====================================================================

lc = 10;
while lc > 0
    lc = lc - 1;
end
assert(strcmp(__inferred_type_str(lc), 'Number'));
assert(lc == 0);

% =====================================================================
% 38. For loop accumulating into tensor
% =====================================================================

result = zeros(1, 5);
for k = 1:5
    result(k) = k * k;
end
assert(strcmp(__inferred_type_str(result), 'Tensor<?, real>'));
assert(result(3) == 9);

% =====================================================================
% 39. Nested function-like pattern: variable reassigned in nested scope
% =====================================================================

outer = 0;
for i = 1:2
    inner_sum = 0;
    for j = 1:3
        inner_sum = inner_sum + j;
    end
    outer = outer + inner_sum;
end
assert(strcmp(__inferred_type_str(outer), 'Number'));
assert(strcmp(__inferred_type_str(inner_sum), 'Number'));
assert(outer == 12);

% =====================================================================
% 40. Empty tensor initialization then loop fill
% =====================================================================

vec = zeros(1, 4);
for k = 1:4
    vec(k) = k;
end
assert(strcmp(__inferred_type_str(vec), 'Tensor<?, real>'));
assert(vec(4) == 4);

disp('SUCCESS');
