% Test log2 two-output form (frexp decomposition)
% [F, E] = log2(X) returns mantissa F and exponent E such that X = F .* 2.^E
% where 0.5 <= abs(F) < 1

% Basic cases
[f, e] = log2(8);
assert(f == 0.5);
assert(e == 4);
assert(f * 2^e == 8);

[f, e] = log2(1);
assert(f == 0.5);
assert(e == 1);
assert(f * 2^e == 1);

[f, e] = log2(12);
assert(f == 0.75);
assert(e == 4);
assert(f * 2^e == 12);

% Verify single-output form still works
assert(abs(log2(8) - 3) < 1e-10);

disp('SUCCESS');
