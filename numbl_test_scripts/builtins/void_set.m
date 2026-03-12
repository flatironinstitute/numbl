% Test that set() doesn't produce output (it's a void function in MATLAB)
% In MATLAB, set(h, prop, val) returns nothing, so even without a semicolon
% no 'ans' is displayed.

h = gcf;

% These should NOT produce any 'ans = ...' output
set(h, 'Name', 'test')
set(h, 'Visible', 'off')

disp('SUCCESS');
