% Tests that $rt.share() is correctly generated when variables
% change type through control flow. These test RUNTIME correctness,
% not just type inference. Incorrect sharing causes aliasing bugs
% where modifying one variable accidentally modifies another.

% =====================================================================
% S1. Basic aliasing prevention: tensor assignment must share
% =====================================================================

a = [1 2 3];
b = a;
b(1) = 99;
assert(a(1) == 1, 'a should not be aliased to b');

% =====================================================================
% S2. Aliasing through loop swap (the chebfun regression pattern)
%     If Pm2 = Pm1 doesn't share, then modifying Pm2 via arithmetic
%     would corrupt Pm1's data.
% =====================================================================

Pm1 = [10 20 30];
Pm2 = 1;
for k = 1:2
    P = Pm1 + k;     % uses Pm1
    Pm2 = Pm1;       % MUST share (Pm1 is tensor)
    Pm1 = P;
end
% Pm2 should be the PREVIOUS Pm1, not the current one
assert(Pm2(1) == 11, 'Pm2 should be copy of Pm1 from iter 1');
assert(Pm1(1) == 13, 'Pm1 should be P from iter 2');

% =====================================================================
% S3. Sharing in conditional branch
% =====================================================================

src = [1 2 3 4 5];
if true
    dst = src;
else
    dst = 0;
end
dst(1) = 999;
assert(src(1) == 1, 'src should not be modified');

% =====================================================================
% S4. Sharing when type is Unknown after control flow
% =====================================================================

v4 = [1 2 3];
if true
    v4 = 42;
else
    v4 = [4 5 6];
end
% v4 is Unknown here. If we assign it, share must be called for safety.
v4_copy = v4;
% v4 is actually 42 (Number) so no aliasing issue in this runtime path,
% but the compiler must generate share for the Unknown case.
assert(v4 == 42);

% =====================================================================
% S5. Tensor created inside loop, assigned to outer variable
% =====================================================================

outer5 = [];
for k = 1:3
    inner5 = [k, k*2, k*3];
    outer5 = inner5;
end
inner5(1) = 999;
% outer5 should be independent of inner5 after sharing
assert(outer5(1) == 3, 'outer5 should not alias inner5');

% =====================================================================
% S6. Three-variable rotation: each assignment must share
% =====================================================================

r_a = [1 2];
r_b = [3 4];
r_c = [5 6];
for k = 1:3
    r_tmp = r_a;
    r_a = r_b;
    r_b = r_c;
    r_c = r_tmp;
end
% After 3 rotations: a=original, b=original, c=original
assert(r_a(1) == 1 && r_a(2) == 2, 'r_a should be [1 2]');
assert(r_b(1) == 3 && r_b(2) == 4, 'r_b should be [3 4]');
assert(r_c(1) == 5 && r_c(2) == 6, 'r_c should be [5 6]');
% Verify no aliasing
r_a(1) = 999;
assert(r_b(1) == 3, 'r_b should not alias r_a');
assert(r_c(1) == 5, 'r_c should not alias r_a');

% =====================================================================
% S7. Assignment from function that returns tensor
% =====================================================================

z7 = zeros(1, 5);
for k = 1:5
    z7(k) = k;
end
z7_copy = z7;
z7(1) = 999;
assert(z7_copy(1) == 1, 'z7_copy should not alias z7');

% =====================================================================
% S8. While loop with tensor swap
% =====================================================================

w_a = [1 2 3];
w_b = [4 5 6];
wi = 0;
while wi < 3
    wi = wi + 1;
    w_tmp = w_a;
    w_a = w_b;
    w_b = w_tmp;
end
% 3 swaps: odd number, so a and b are swapped from original
assert(w_a(1) == 4, 'w_a should be [4 5 6]');
assert(w_b(1) == 1, 'w_b should be [1 2 3]');
w_a(1) = 999;
assert(w_b(1) == 1, 'w_b should not alias w_a');

% =====================================================================
% S9. Tensor assigned inside try/catch
% =====================================================================

tc9 = [1 2 3];
try
    tc9_backup = tc9;
catch
    tc9_backup = [0 0 0];
end
tc9(1) = 999;
assert(tc9_backup(1) == 1, 'tc9_backup should not alias tc9');

% =====================================================================
% S10. Tensor assigned inside switch
% =====================================================================

sw_src = [10 20 30];
switch 2
    case 1
        sw_dst = [0 0 0];
    case 2
        sw_dst = sw_src;
    otherwise
        sw_dst = [1 1 1];
end
sw_src(1) = 999;
assert(sw_dst(1) == 10, 'sw_dst should not alias sw_src');

% =====================================================================
% S11. Legendre recurrence with derivative tracking
%      (Exact pattern from chebfun legpts that caused NaN regression)
% =====================================================================

x11 = linspace(0, 1, 8);
Pm2_11 = 1;
Pm1_11 = x11;
PPm2_11 = 0;
PPm1_11 = 1;

for k = 1:3
    P11 = ((2*k+1)*Pm1_11.*x11 - k*Pm2_11) / (k+1);
    Pm2_11 = Pm1_11;
    Pm1_11 = P11;
    PP11 = ((2*k+1)*(Pm2_11 + x11.*PPm1_11) - k*PPm2_11) / (k+1);
    PPm2_11 = PPm1_11;
    PPm1_11 = PP11;
end

% Verify P_4(x) = (35x^4 - 30x^2 + 3)/8
P4_exact = (35*x11.^4 - 30*x11.^2 + 3) / 8;
assert(max(abs(P11 - P4_exact)) < 1e-10, 'P4 correctness');

% Verify P_4'(x) = (140x^3 - 60x)/8 = (35x^3 - 15x)/2
PP4_exact = (140*x11.^3 - 60*x11) / 8;
assert(max(abs(PP11 - PP4_exact)) < 1e-10, 'P4 derivative correctness');

% Verify no aliasing between Pm2 and Pm1
Pm2_11_orig = Pm2_11(1);
Pm1_11(1) = 999;
assert(Pm2_11(1) == Pm2_11_orig, 'Pm2 should not alias Pm1');

% =====================================================================
% S12. While+for nested Newton iteration
%      (Same pattern that caused the diskfun regression)
% =====================================================================

x12 = [0.8; 0.3];
n12 = 4;
Pm2_12 = 1;
Pm1_12 = x12;
PPm2_12 = 0;
PPm1_12 = 1;
dx12 = inf;
ctr12 = 0;

while norm(dx12, inf) > eps && ctr12 < 10
    ctr12 = ctr12 + 1;
    for k = 1:n12-1
        P12 = ((2*k+1)*Pm1_12.*x12 - k*Pm2_12) / (k+1);
        Pm2_12 = Pm1_12;
        Pm1_12 = P12;
        PP12 = ((2*k+1)*(Pm2_12 + x12.*PPm1_12) - k*PPm2_12) / (k+1);
        PPm2_12 = PPm1_12;
        PPm1_12 = PP12;
    end
    dx12 = -P12./PP12;
    x12 = x12 + dx12;
    Pm2_12 = 1;
    Pm1_12 = x12;
    PPm2_12 = 0;
    PPm1_12 = 1;
end

assert(~any(isnan(x12)), 'Newton should not produce NaN');
assert(ctr12 > 0 && ctr12 <= 10, 'Newton should converge');
P4_roots = (35*x12.^4 - 30*x12.^2 + 3) / 8;
assert(max(abs(P4_roots)) < 1e-10, 'Should converge to Legendre roots');

disp('SUCCESS');
