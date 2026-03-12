% Test that calling a function with unassigned output variable raises an error

error_raised = false;
try
    a = helper1(4);
catch e
    error_raised = true;
end
assert(error_raised, 'Expected error for unassigned output variable');

disp('SUCCESS');

function y = helper1(x)
end
