% Test fast paths for single-row and single-column tensor indexing.

%% Single row extraction c(k,:)
A = reshape(1:24, 4, 6);
row2 = A(2,:);
assert(isequal(row2, [2 6 10 14 18 22]), 'row extraction values');
assert(isequal(size(row2), [1 6]), 'row extraction shape');

%% Single column extraction c(:,k)
col3 = A(:,3);
assert(isequal(col3, [9; 10; 11; 12]), 'col extraction values');
assert(isequal(size(col3), [4 1]), 'col extraction shape');

%% First and last rows
assert(isequal(A(1,:), [1 5 9 13 17 21]), 'first row');
assert(isequal(A(4,:), [4 8 12 16 20 24]), 'last row');

%% First and last cols
assert(isequal(A(:,1), [1;2;3;4]), 'first col');
assert(isequal(A(:,6), [21;22;23;24]), 'last col');

%% Complex matrix
B = reshape((1:12) + 1i*(13:24), 3, 4);
row1 = B(1,:);
assert(abs(real(row1(1)) - 1) < 1e-10, 'complex row real');
assert(abs(imag(row1(1)) - 13) < 1e-10, 'complex row imag');
assert(isequal(size(row1), [1 4]), 'complex row shape');

col2 = B(:,2);
assert(abs(real(col2(1)) - 4) < 1e-10, 'complex col real');
assert(isequal(size(col2), [3 1]), 'complex col shape');

%% Row extraction in a loop (the fevalm/Clenshaw pattern)
C = randn(71, 100);
total = 0;
for k = 1:71
    row = C(k,:);
    total = total + sum(row);
end
assert(abs(total - sum(C(:))) < 1e-8, 'loop row extraction sum');

%% Out of bounds should error
try
    A(0,:);
    assert(false, 'should have errored');
catch
end

try
    A(5,:);
    assert(false, 'should have errored');
catch
end

try
    A(:,0);
    assert(false, 'should have errored');
catch
end

try
    A(:,7);
    assert(false, 'should have errored');
catch
end

disp('SUCCESS');
