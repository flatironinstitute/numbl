% `clear` command — actually remove variables from the workspace.
%
% MATLAB form (command-form, args parsed as strings):
%   clear            — clear all variables in current scope
%   clear var1 var2  — clear named variables
%
% Reading a cleared variable must produce an "Unrecognized variable"
% error, exactly like reading an undefined name.

% --- 1) Single named variable
x = 5;
clear x
threw = false;
try
    x;
catch
    threw = true;
end
assert(threw, '1: clear x must remove x');

% --- 2) Multiple named variables
a = 1; b = 2; c = 3;
clear a b
threw_a = false;
try
    a;
catch
    threw_a = true;
end
threw_b = false;
try
    b;
catch
    threw_b = true;
end
assert(threw_a && threw_b, '2: clear a b must remove a and b');
assert(c == 3, '2: clear a b must NOT touch c');

% --- 3) Re-assignment after clear is fine — variable comes back.
v = 7;
clear v
v = 42;
assert(v == 42, '3: re-assignment after clear works');

% --- 4) Bare `clear` (no args) clears all locals in current scope.
%      Re-introduce some vars first so we have something to clear.
p = 'one';
q = [1 2 3];
clear
threw_p = false;
try
    p;
catch
    threw_p = true;
end
threw_q = false;
try
    q;
catch
    threw_q = true;
end
assert(threw_p && threw_q, '4: bare clear removes all locals');

% --- 5) Clearing a non-existent name is a silent no-op (matches MATLAB).
clear no_such_var_anywhere

disp('SUCCESS')
