% Regression: JS-JIT logical-mask indexed store a(mask)=rhs.
%
% Valid writes (matching length, scalar broadcast) must produce the correct
% values under the JIT. A length-mismatched write must RAISE "Subscripted
% assignment dimension mismatch" — before the fix the JS-JIT single-slot
% LogicalMask store had no length check (unlike the Range store and the C
% path), so a short RHS silently padded with NaN and a long RHS silently
% truncated.
%
% The mask/RHS are loop inputs and the array literal is in-loop, which is
% the shape the jit-loop executor compiles (assert_jit pins engagement).

% Matching-length tensor RHS.
m = logical([1 1 1 0 0 0]);
vals = [7 8 9];
total = 0;
for k = 1:30
    %!numbl:assert_jit
    a = [1 2 3 4 5 6];
    a(m) = vals;                 % -> [7 8 9 4 5 6]
    total = total + sum(a);
end
assert(total == 30 * 39, 'matching-length mask write');

% Scalar-broadcast RHS.
m2 = logical([1 0 1 0]);
total2 = 0;
for k = 1:30
    %!numbl:assert_jit
    b = [1 2 3 4];
    b(m2) = 99;                  % -> [99 2 99 4]
    total2 = total2 + sum(b);
end
assert(total2 == 30 * 204, 'scalar-broadcast mask write');

% A length-mismatched mask store (3 slots, 2 values) must error rather than
% silently corrupt. Same JIT-eligible shape as above.
threw = false;
try
    mbad = logical([1 1 1 0 0 0]);
    short = [7 8];
    for k = 1:30
        d = [1 2 3 4 5 6];
        d(mbad) = short;
    end
catch
    threw = true;
end
assert(threw, 'length-mismatched logical-mask write must error');

disp('SUCCESS')
