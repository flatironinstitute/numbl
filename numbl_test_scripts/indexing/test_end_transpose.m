% A transpose quote immediately after the `end` index keyword must lex as a
% transpose, not as the start of a string. numbl's isValueToken set listed
% True/False but omitted End, so `A(end')` raised "Invalid token: '''".

%!numbl:assert_jit c

A = [10 20 30];
assert(isequal(A(end'), 30), 'A(end'') should be A(end) = 30');

B = [1 2; 3 4];
% linear index: column-major, end = 4, B(4) = 4
assert(isequal(B(end'), 4), 'B(end'') linear index');

% end' inside an expression
assert(isequal(A(end' - 1), 20), 'A(end''-1)');

disp('SUCCESS');
