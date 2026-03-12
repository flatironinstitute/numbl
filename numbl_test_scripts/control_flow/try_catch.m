% try / catch

% Catch a runtime error
caught = false;
try
  x = 1 / 0;  % inf, not an error in MATLAB
  error('forced error');
catch e
  caught = true;
end
assert(caught)

% Error message accessible
try
  error('my message');
catch e
  assert(~isempty(strfind(e.message, 'my message')))
end

% No error: catch not triggered
result = 0;
try
  result = 42;
catch
  result = -1;
end
assert(result == 42)

% error with sprintf-style formatting
try
  error('value is %d', 5);
catch e
  assert(~isempty(strfind(e.message, '5')))
end

% try without catch: error is silently ignored
result2 = 0;
try
  result2 = 1;
  error('ignored error');
  result2 = 2;
end
assert(result2 == 1)

% try without catch: no error
result3 = 0;
try
  result3 = 42;
end
assert(result3 == 42)

disp('SUCCESS')
