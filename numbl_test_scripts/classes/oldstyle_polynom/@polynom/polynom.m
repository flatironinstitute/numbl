function p = polynom(c)
%POLYNOM old-style polynomial class (coeffs high->low order)
if nargin == 0
  p.c = [];
  p = class(p, 'polynom');
elseif isa(c, 'polynom')
  p = c;
else
  p.c = c(:).';
  p = class(p, 'polynom');
end
