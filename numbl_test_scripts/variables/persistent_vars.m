% Test persistent variables retain values across function calls

% Test 1: Basic persistent counter
assert(counter() == 1);
assert(counter() == 2);
assert(counter() == 3);

% Test 2: Multiple persistent variables in the same function
[a, b] = dual_persistent();
assert(a == 1 && b == 10);
[a, b] = dual_persistent();
assert(a == 2 && b == 20);

% Test 3: Persistent in static method (chebfunpref pattern)
s = PrefHolder_.getDefaults('get');
assert(isstruct(s));
assert(s.alpha == 10);
assert(s.beta == 20);

disp('SUCCESS')

function out = counter()
  persistent count
  if isempty(count)
    count = 0;
  end
  count = count + 1;
  out = count;
end

function [a, b] = dual_persistent()
  persistent x
  persistent y
  if isempty(x)
    x = 0;
    y = 0;
  end
  x = x + 1;
  y = y + 10;
  a = x;
  b = y;
end
