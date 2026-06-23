function p = superiorpoly(v)
%SUPERIORPOLY Minimal old-style (pre-classdef) class using superiorto.
  superiorto('double');
  if nargin == 0
    v = [];
  end
  s.v = v(:);
  p = class(s, 'superiorpoly');
end
