% A script invoked by name whose file also defines local functions.
%
% numbl resolved `<name>;` by taking the first `function` in the file and
% calling it. For a script with trailing local functions that is a local
% function, invoked with none of its arguments bound, so it failed on the
% first parameter read. A file is a function file only when its *first*
% statement is a function declaration; otherwise it is a script.

script_with_locals_;
assert(exist('sl_x', 'var') == 1, 'the script ran in this workspace');
assert(sl_x == 30, 'its local function received its argument');

% The rule it must not break: a function file whose declared name differs from
% its file name is still reached through the file name.
assert(oddly_named_file_(41) == 42, 'file name wins over function name');

disp('SUCCESS')
