% Test that assigning empty RHS when indices select zero elements is a no-op
% X(emptyRange, someIdx) = zeros(0,n) should not error or delete

X = [1 2 3; 4 5 6; 7 8 9];

% Empty row index with matching empty RHS: no-op
m = 1;
evenModes = [1 2];
sub = X(2:2:m, evenModes);    % 0x2 empty
result = bsxfun(@minus, sub, 2*sum(sub, 1));  % 0x2 empty
X(2:2:m, evenModes) = result;
assert(isequal(X, [1 2 3; 4 5 6; 7 8 9]), 'empty row range assign should be no-op');

% Another pattern: direct zeros(0,n) assignment
Y = magic(4);
Y(5:5:1, [1 3]) = zeros(0, 2);
assert(isequal(Y, magic(4)), 'zeros(0,n) assign with empty rows should be no-op');

% Empty column range
Z = [1 2; 3 4; 5 6];
Z([1 3], 3:3:1) = zeros(2, 0);
assert(isequal(Z, [1 2; 3 4; 5 6]), 'empty col range assign should be no-op');

disp('SUCCESS');
