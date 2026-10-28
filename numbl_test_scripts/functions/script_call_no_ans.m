% Running a script by name produces no value: it neither sets nor displays
% `ans` in the caller, and calling it with arguments or an output errors.

clear ans
script_no_ans_helper_
assert(scna_z == 6);
assert(~exist('ans', 'var'));

script_no_ans_helper_;
assert(~exist('ans', 'var'));

try
  x = script_no_ans_helper_; %#ok<NASGU>
  error('expected an error');
catch e
  assert(strcmp(e.identifier, 'MATLAB:scriptNotAFunction'));
  assert(contains(e.message, 'Execution of script script_no_ans_helper_ as a function is not supported'));
end

try
  script_no_ans_helper_(3);
  error('expected an error');
catch e
  assert(strcmp(e.identifier, 'MATLAB:scriptNotAFunction'));
end

disp('SUCCESS')
