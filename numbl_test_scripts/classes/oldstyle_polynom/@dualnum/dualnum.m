function d = dualnum(v)
if nargin == 0, v = 0; end
d.v = v;
d = class(d, 'dualnum');
