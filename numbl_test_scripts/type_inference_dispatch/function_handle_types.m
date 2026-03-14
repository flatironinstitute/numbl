% Test function handle type inference
% Each section tests a specific improvement

% =====================================================================
% AnonFunc body type inference
% =====================================================================

% --- Anonymous function returning constant type ---
f1 = @(x) true;
assert(strcmp(__inferred_type_str(f1), 'Function<Unknown, Boolean>'));

f2 = @(x) 'hello';
assert(strcmp(__inferred_type_str(f2), 'Function<Unknown, Char>'));

% --- Anonymous function with unknown param type ---
f3 = @(x) x + 1;
% x is Unknown, so x+1 is Unknown
assert(strcmp(__inferred_type_str(f3), 'Function<Unknown, Unknown>'));

% =====================================================================
% FuncHandle type inference
% =====================================================================

g1 = @sin;
% sin with unknown arg → function with unknown return
assert(strcmp(__inferred_type_str(g1), 'Function<, Unknown>'));

% =====================================================================
% Calling function handle propagates return type
% =====================================================================

% Return type is known (constant Boolean) → propagates through call
f4 = @(x) true;
result1 = f4(42);
assert(strcmp(__inferred_type_str(result1), 'Boolean'));
assert(result1 == true);

% Return type is known (Char) → propagates through call
f5 = @(x) 'yes';
result2 = f5(42);
assert(strcmp(__inferred_type_str(result2), 'Char'));
assert(strcmp(result2, 'yes'));

% =====================================================================
% arrayfun return type (when function return type is known)
% =====================================================================

% Anonymous function with known Boolean return type
y1 = arrayfun(@(x) true, [1 2 3]);
assert(strcmp(__inferred_type_str(y1), 'Tensor<?, real, logical>'));

% Anonymous function with unknown return type (x*2 where x is Unknown)
y2 = arrayfun(@(x) x * 2, [1 2 3]);
assert(isequal(y2, [2 4 6]));

% =====================================================================
% cellfun return type (when function return type is known)
% =====================================================================

y3 = cellfun(@(x) true, {1, 2, 3});
assert(strcmp(__inferred_type_str(y3), 'Tensor<?, real, logical>'));

% =====================================================================
% Runtime correctness
% =====================================================================

f6 = @(x) x * 2 + 1;
assert(f6(3) == 7);
assert(f6(10) == 21);

disp('SUCCESS')
