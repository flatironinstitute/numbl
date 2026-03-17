% Test that for loops handle cross-iteration type changes correctly.
% Legendre polynomial recurrence: Pm2 starts as scalar (1) but becomes
% a tensor on iteration 2+ via Pm2 = Pm1. The loop body arithmetic
% (k*Pm2, Pm1.*x, etc.) must work correctly regardless.

function for_loop_type_change()

% --- Case 1: for loop only ---
x = linspace(0, 1, 10);
Pm1 = x;            % Tensor
Pm2 = 1;            % Number - becomes Tensor on iter 2
PPm1 = ones(1, 10); % Tensor
PPm2 = 0;           % Number - becomes Tensor on iter 2

% Legendre recurrence: P_{k+1} = ((2k+1)*x*P_k - k*P_{k-1})/(k+1)
for k = 1:3
    P = ((2*k+1)*Pm1.*x - k*Pm2) / (k+1);
    Pm2 = Pm1;
    Pm1 = P;
    PP = ((2*k+1)*(Pm2 + x.*PPm1) - k*PPm2) / (k+1);
    PPm2 = PPm1;
    PPm1 = PP;
end

% All should be tensors after the loop
assert(numel(P) == 10, 'P should be a tensor');
assert(numel(PP) == 10, 'PP should be a tensor');
assert(numel(Pm2) == 10, 'Pm2 should be a tensor');
assert(numel(PPm2) == 10, 'PPm2 should be a tensor');
assert(~any(isnan(P)), 'P should not contain NaN');
assert(~any(isnan(PP)), 'PP should not contain NaN');

% Verify numerical correctness: after k=1:3, P = P_4(x) = (35x^4 - 30x^2 + 3)/8
P4_exact = (35*x.^4 - 30*x.^2 + 3) / 8;
assert(max(abs(P - P4_exact)) < 1e-10, 'Legendre P4 should be correct');

% --- Case 2: while loop containing for loop (Newton iteration for ---
% Legendre roots, same pattern as chebfun's legpts/rec).
% Variables are re-initialized to scalars each while iteration, but the
% for loop turns them into tensors. Both loops must handle this correctly.
n = 4;
x2 = [0.8; 0.3];
Pm2 = 1;
Pm1 = x2;
PPm2 = 0;
PPm1 = 1;
dx = inf;
counter = 0;

while norm(dx, inf) > eps && counter < 10
    counter = counter + 1;
    for k = 1:n-1
        P = ((2*k+1)*Pm1.*x2-k*Pm2)/(k+1);
        Pm2 = Pm1;
        Pm1 = P;
        PP = ((2*k+1)*(Pm2+x2.*PPm1)-k*PPm2)/(k+1);
        PPm2 = PPm1;
        PPm1 = PP;
    end
    dx = -P./PP;
    x2 = x2 + dx;
    Pm2 = 1;
    Pm1 = x2;
    PPm2 = 0;
    PPm1 = 1;
end

assert(~any(isnan(x2)), 'Newton iteration should not produce NaN');
assert(counter > 0 && counter <= 10, 'Newton should converge');
% x2 should converge to roots of P_4 (Legendre polynomial of degree 4)
% Verify by evaluating P_4 at the computed roots
P4_at_roots = (35*x2.^4 - 30*x2.^2 + 3) / 8;
assert(max(abs(P4_at_roots)) < 1e-10, 'Should converge to Legendre roots');

disp('SUCCESS');
end
