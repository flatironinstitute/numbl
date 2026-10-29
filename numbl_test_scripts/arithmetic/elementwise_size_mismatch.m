% Every element-wise operation raises MATLAB's size-mismatch error for
% operands whose sizes do not broadcast.

a = [1 2 3];
b = [1 2];
ops = {@(x, y) x + y, @(x, y) x - y, @(x, y) x .* y, @(x, y) x ./ y, ...
       @(x, y) x .\ y, @(x, y) x .^ y, @(x, y) x == y, @(x, y) x ~= y, ...
       @(x, y) x < y, @(x, y) x <= y, @(x, y) x > y, @(x, y) x >= y, ...
       @(x, y) x & y, @(x, y) x | y, @(x, y) xor(x, y), @(x, y) mod(x, y), ...
       @(x, y) rem(x, y), @(x, y) atan2(x, y), @(x, y) hypot(x, y), ...
       @(x, y) min(x, y), @(x, y) max(x, y), @(x, y) power(x, y), ...
       @(x, y) plus(x, y), @(x, y) and(x, y)};
operands = {{a, b}, {a * 1i, b}, {a, b * 1i}, {'abc', 'ab'}, ...
            {[1 2 3], ones(2, 2, 2)}, {zeros(0, 3), zeros(3, 0)}, ...
            {sparse(a), b}};
realOnly = [13:18 24];  % &, |, xor, mod, rem, atan2, and reject complex
floatOnly = [18:21];     % atan2, hypot, min and max reject a char second operand
for k = 1:numel(ops)
  for j = 1:numel(operands)
    if (ismember(k, realOnly) && ismember(j, [2 3])) || ...
       (ismember(k, floatOnly) && j == 4)
      continue;
    end
    x = operands{j}{1};
    y = operands{j}{2};
    threw = false;
    try
      ops{k}(x, y);
    catch e
      threw = true;
      assert(strcmp(e.identifier, 'MATLAB:sizeDimensionsMustMatch'), ...
             sprintf('op %d case %d: %s', k, j, e.identifier));
      assert(strcmp(e.message, ...
             'Arrays have incompatible sizes for this operation.'));
    end
    assert(threw, sprintf('op %d case %d did not throw', k, j));
  end
end

% bsxfun keeps its own error.
try
  bsxfun(@(x, y) x + y, a, b);
  assert(false);
catch e
  assert(strcmp(e.identifier, 'MATLAB:bsxfun:arrayDimensionsMustMatch'));
end

% Singleton dimensions expand; a 1 against a 0 gives 0.
assert(isequal(size(zeros(1, 0) + zeros(2, 1)), [2 0]));
assert(isequal(size(ones(3, 1, 2) .* ones(1, 4)), [3 4 2]));
assert(isequal(atan2(ones(2, 1), ones(1, 3)), atan2(1, 1) * ones(2, 3)));
assert(isequal(mod([5; 7], [2 3]), [1 2; 1 1]));

disp('SUCCESS')
