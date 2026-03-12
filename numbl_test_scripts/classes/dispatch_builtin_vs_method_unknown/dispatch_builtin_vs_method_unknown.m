% Test that class methods take priority over builtins with the same name,
% even when the argument type is Unknown at compile time.
%
% The key scenario: an Unknown value passes through a builtin (max) whose
% return type is incorrectly inferred as Num. Then a subsequent call (sum)
% should still dispatch to the class method, not the builtin.

% Create via helper function so the return type is Unknown at compile time.
s = make_summer(5);

% max(s, 3) should call @Summer/max, returning a Summer.
% But the compiler may infer the return type as Num (from builtin max).
t = max(s, 3);

% sum(t) should call @Summer/sum, not the builtin sum.
% This fails if max's return type was incorrectly inferred as Num.
result = sum(t);

assert(result == 80);

disp('SUCCESS');
