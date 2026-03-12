v = [10, 20, 30, 40, 50];

% Range indexing
sub = v(2:4);
disp(length(sub))
disp(sub(1))
disp(sub(2))
disp(sub(3))

% end keyword
last = v(end);
disp(last)

sub2 = v(3:end);
disp(length(sub2))
disp(sub2(1))

% 2D
A = [1,2,3; 4,5,6; 7,8,9];
row2 = A(2, :);
disp(length(row2))
disp(row2(1))
disp(row2(3))

col3 = A(:, 3);
disp(length(col3))
disp(col3(1))
disp(col3(3))

sub = A(1:2, 2:3);
disp(size(sub, 1))
disp(size(sub, 2))
disp(sub(1,1))
disp(sub(2,2))
