function w = widget(name, val)
if nargin == 0, name = ''; val = []; end
w.name = name;
w.val = val;
w = class(w, 'widget');
