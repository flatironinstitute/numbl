% Functions whose names collide with Object.prototype properties
% (constructor, toString, valueOf, hasOwnProperty) must not be
% misinterpreted as plot intrinsics or other built-in dispatches.

assert(constructor(3) == 9);
assert(toString(5) == 10);
assert(valueOf(7) == 14);
assert(hasOwnProperty(2) == 4);

disp('SUCCESS')

function y = constructor(x)
  y = x * x;
end

function y = toString(x)
  y = x * 2;
end

function y = valueOf(x)
  y = x * 2;
end

function y = hasOwnProperty(x)
  y = x * 2;
end
