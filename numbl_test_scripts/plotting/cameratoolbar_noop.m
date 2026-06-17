% cameratoolbar is accepted as a no-op (numbl has no interactive camera
% toolbar). It must not error in any of its documented forms.

cameratoolbar
cameratoolbar('show');
cameratoolbar('SetMode', 'orbit');
cameratoolbar('SetCoordSys', 'y');
cameratoolbar(1, 'show');           % leading figure handle
cameratoolbar('resetcamera');

% Output forms return a (dummy) value rather than erroring.
tb = cameratoolbar('SetMode', 'orbit');
assert(~isempty(tb), 'returns a value when an output is requested');
v = cameratoolbar('GetVisible');
assert(~isempty(v), 'GetVisible returns a value');

disp('SUCCESS');
