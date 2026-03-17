% Adversarial tests for flow-dependent type inference.
% These target known brittleness in the implementation.

% =====================================================================
% A1. Three-way fixpoint: requires >1 iteration to stabilize
%     a=Number, b=Tensor, c=Char. Rotate each iteration.
%     After fixpoint: all should be Unknown.
%     Runtime: verify values are correct after rotation.
% =====================================================================

a3 = 10;
b3 = [1 2 3];
c3 = 'hi';
for k = 1:6
    old_a = a3;
    a3 = b3;
    b3 = c3;
    c3 = old_a;
end
% 6 rotations = 2 full cycles, back to original assignment
assert(a3 == 10);
assert(numel(b3) == 3);
assert(strcmp(c3, 'hi'));

% =====================================================================
% A2. Conditional swap inside loop - type depends on branch taken
% =====================================================================

cs_a = 1;
cs_b = [1 2];
for k = 1:4
    if mod(k, 2) == 0
        tmp_cs = cs_a;
        cs_a = cs_b;
        cs_b = tmp_cs;
    end
end
% k=2: swap (a=tensor, b=scalar), k=4: swap back (a=scalar, b=tensor)
assert(cs_a == 1);
assert(numel(cs_b) == 2);

% =====================================================================
% A3. Variable that changes type via function return value in loop
% =====================================================================

fv = 0;
for k = 1:3
    fv = fv + k;
end
assert(fv == 6);
assert(strcmp(__inferred_type_str(fv), 'Number'));

% =====================================================================
% A4. Nested loop where inner loop changes type of outer variable
%     Outer var starts as Number, inner loop makes it tensor.
% =====================================================================

ov = 0;
for i = 1:2
    for j = 1:2
        ov = ov + 1;
    end
end
assert(strcmp(__inferred_type_str(ov), 'Number'));
assert(ov == 4);

% Now test where inner loop actually changes type
ov2 = 1;
for i = 1:2
    for j = 1:1
        ov2 = [ov2, j];  % changes from Number to Tensor
    end
end
assert(strcmp(__inferred_type_str(ov2), 'Unknown'));

% =====================================================================
% A5. While loop where the condition uses a variable assigned in body
%     with type that changes (Number -> Unknown)
% =====================================================================

wc_val = 10;
wc_ctr = 0;
while wc_val > 0
    wc_val = wc_val - 3;
    wc_ctr = wc_ctr + 1;
end
assert(strcmp(__inferred_type_str(wc_val), 'Number'));
assert(strcmp(__inferred_type_str(wc_ctr), 'Number'));
assert(wc_val == -2);
assert(wc_ctr == 4);

% =====================================================================
% A6. Deep nesting stress: 4 levels of control flow
% =====================================================================

d4 = 0;
for i = 1:2
    for j = 1:2
        if true
            for kk = 1:2
                if true
                    d4 = d4 + 1;
                end
            end
        end
    end
end
assert(strcmp(__inferred_type_str(d4), 'Number'));
assert(d4 == 8);

% =====================================================================
% A7. Variable assigned in BOTH branches of if INSIDE a for loop
%     where one branch changes type
% =====================================================================

ab = 0;
for k = 1:4
    if mod(k, 2) == 0
        ab = ab + 1;  % Number
    else
        ab = 'odd';   % Char
    end
end
% ab is assigned in both branches: Number in one, Char in other
% After if: Unknown. After loop join with pre-loop(Number): Unknown
assert(strcmp(__inferred_type_str(ab), 'Unknown'));

% =====================================================================
% A8. Matrix accumulation in loop - type should stay Tensor
% =====================================================================

M = zeros(3,3);
for k = 1:3
    M(k,k) = k;
end
assert(strcmp(__inferred_type_str(M), 'Tensor<?, real>'));
assert(M(2,2) == 2);

% =====================================================================
% A9. Variable used in expression on RHS of its own reassignment
%     where the type changes
% =====================================================================

self = 1;
for k = 1:3
    self = self + k;
end
assert(strcmp(__inferred_type_str(self), 'Number'));
assert(self == 7);

% Now with type change
self2 = 1;
self2 = [self2, 2, 3];  % Number -> Tensor
assert(strcmp(__inferred_type_str(self2), 'Tensor<?, real>'));
assert(numel(self2) == 3);

% =====================================================================
% A10. Switch inside while loop
% =====================================================================

sw_w = 0;
sw_i = 1;
while sw_i <= 4
    switch mod(sw_i, 3)
        case 0
            sw_w = sw_w + 10;
        case 1
            sw_w = sw_w + 1;
        otherwise
            sw_w = sw_w + 100;
    end
    sw_i = sw_i + 1;
end
assert(strcmp(__inferred_type_str(sw_w), 'Number'));
assert(sw_w == 112);

% =====================================================================
% A11. Try/catch inside while loop where catch changes type
% =====================================================================

tc_w = 0;
tc_i = 0;
while tc_i < 3
    tc_i = tc_i + 1;
    try
        tc_w = tc_w + tc_i;
    catch
        tc_w = 'error';
    end
end
assert(strcmp(__inferred_type_str(tc_w), 'Unknown'));
assert(tc_w == 6);  % no errors actually thrown

% =====================================================================
% A12. Variable only assigned in catch block
% =====================================================================

catch_only = 'safe';
try
    % no error
catch
    catch_only = 42;
end
% Pre-branch: Char, try-branch: unchanged(Char), catch-branch: Number
% Join try+catch: Char vs Number = Unknown? Or Char?
% TryCatch joins try-post and catch-post with pre-branch included
assert(strcmp(__inferred_type_str(catch_only), 'Unknown'));

% =====================================================================
% A13. Multiple loops in sequence, each narrowing back to precise type
% =====================================================================

ml = 'start';
assert(strcmp(__inferred_type_str(ml), 'Char'));

ml = 0;
for k = 1:5
    ml = ml + 1;
end
assert(strcmp(__inferred_type_str(ml), 'Number'));
assert(ml == 5);

ml = [1 2 3];
for k = 1:3
    ml(k) = ml(k) * 2;
end
assert(strcmp(__inferred_type_str(ml), 'Tensor<?, real>'));
assert(ml(2) == 4);

% =====================================================================
% A14. Verify $rt.share is correctly generated for tensor assignments
%      in loops (the actual chebfun regression pattern)
% =====================================================================

x14 = linspace(0, 1, 10);
Pm2_14 = 1;            % Number (scalar)
Pm1_14 = x14;          % Tensor
PPm2_14 = 0;           % Number
PPm1_14 = ones(1, 10); % Tensor

for k = 1:3
    P14 = ((2*k+1)*Pm1_14.*x14 - k*Pm2_14) / (k+1);
    Pm2_14 = Pm1_14;   % this MUST use $rt.share since Pm1_14 could be tensor
    Pm1_14 = P14;
    PP14 = ((2*k+1)*(Pm2_14 + x14.*PPm1_14) - k*PPm2_14) / (k+1);
    PPm2_14 = PPm1_14;
    PPm1_14 = PP14;
end

% After loop: all should be tensors, no NaN
assert(numel(P14) == 10, 'P14 should be tensor');
assert(numel(PP14) == 10, 'PP14 should be tensor');
assert(numel(Pm2_14) == 10, 'Pm2_14 should be tensor');
assert(numel(PPm2_14) == 10, 'PPm2_14 should be tensor');
assert(~any(isnan(P14)), 'P14 no NaN');
assert(~any(isnan(PP14)), 'PP14 no NaN');

% Verify P4(x) = (35x^4 - 30x^2 + 3)/8
P4_exact14 = (35*x14.^4 - 30*x14.^2 + 3) / 8;
assert(max(abs(P14 - P4_exact14)) < 1e-10, 'P4 correctness');

% =====================================================================
% A15. For loop where body has nested if that changes var type,
%      then loop body re-assigns to original type
% =====================================================================

recover = 0;
for k = 1:3
    if k == 2
        recover = 'temp';
    end
    recover = k;  % always Number at end of body
end
% Pre-loop: Number, body ends: Number -> join: Number
assert(strcmp(__inferred_type_str(recover), 'Number'));
assert(recover == 3);

% =====================================================================
% A16. Boolean operations in loops
% =====================================================================

flag16 = true;
for k = 1:5
    flag16 = flag16 && (k < 10);
end
assert(strcmp(__inferred_type_str(flag16), 'Boolean'));
assert(flag16 == true);

% =====================================================================
% A17. Empty tensor growing in while loop
% =====================================================================

grow = [];
gi = 0;
while gi < 5
    gi = gi + 1;
    grow = [grow, gi];
end
assert(strcmp(__inferred_type_str(grow), 'Tensor<?, real>'));
assert(numel(grow) == 5);
assert(grow(3) == 3);

% =====================================================================
% A18. Struct field assignment inside loop
% =====================================================================

st18 = struct();
st18.sum = 0;
for k = 1:3
    st18.sum = st18.sum + k;
end
assert(st18.sum == 6);

% =====================================================================
% A19. For-for nested with shared accumulator and separate counter
% =====================================================================

acc19 = 0;
for i = 1:3
    for j = 1:i
        acc19 = acc19 + 1;
    end
end
assert(strcmp(__inferred_type_str(acc19), 'Number'));
assert(acc19 == 6);  % 1 + 2 + 3

% =====================================================================
% A20. Fibonacci-style recurrence (two-variable swap)
% =====================================================================

fib_a = 0;
fib_b = 1;
for k = 1:10
    fib_c = fib_a + fib_b;
    fib_a = fib_b;
    fib_b = fib_c;
end
assert(strcmp(__inferred_type_str(fib_a), 'Number'));
assert(strcmp(__inferred_type_str(fib_b), 'Number'));
assert(fib_b == 89);  % fib(11)

disp('SUCCESS');
