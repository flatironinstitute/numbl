function r = plus(a, b)
%PLUS Overloaded +; works with the object on either side.
  if isa(a, 'superiorpoly')
    obj = a; other = b;
  else
    obj = b; other = a;
  end
  r = sum(obj.v) + other;
end
