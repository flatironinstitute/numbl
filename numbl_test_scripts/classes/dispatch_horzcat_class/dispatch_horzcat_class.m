% Test that builtin dispatch works correctly when a class instance
% is created via horzcat (matrix literal syntax [a, b]).
% The resulting object should dispatch qr() to the class method,
% not the builtin qr.

a = QrObj_(3);
b = QrObj_(4);
c = [a, b];  % horzcat -> QrObj_(7)
result = qr(c);
assert(result == 70, 'Expected qr([a,b]) to dispatch to class method');

fprintf('SUCCESS\n');
