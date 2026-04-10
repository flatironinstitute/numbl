% Test mexext builtin — returns the mex-file extension for the current
% platform.

e = mexext;
assert(e == 'numbl.js');

disp('SUCCESS');
