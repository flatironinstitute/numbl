function m = cell2d(n, recurse)
if nargin < 2, recurse = true; end
m.ndim = n;
m.val = n * 10;
if recurse && n > 1
  for iside = 1:2
    m.boundary(iside) = cell2d(n-1, false);   % grow from undefined field
    m.boundary(iside).val = 900 + iside;        % chained assign into element
  end
else
  m.boundary = [];
end
m = class(m, 'cell2d');
