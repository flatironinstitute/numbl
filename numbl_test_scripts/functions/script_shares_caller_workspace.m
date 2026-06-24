% A script invoked by name runs in the caller's workspace (MATLAB semantics):
% variables it assigns are visible to the caller afterwards.

make_vars_script_;

assert(exist('sv_alpha', 'var') == 1, 'script did not create sv_alpha in caller workspace');
assert(sv_alpha == 11);
assert(isequal(sv_beta, [2 4 6]));

disp('SUCCESS')
