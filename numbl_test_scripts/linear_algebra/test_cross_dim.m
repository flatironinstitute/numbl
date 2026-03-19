% Test cross product with dim argument and auto-dim detection

% --- cross(A,B) auto-detect first dimension of size 3 ---

% 2x3 matrices: no dim is 3 except dim 2 (size 3), so cross along dim 2
A = [1 0 0; 0 1 0];
B = [0 1 0; 0 0 1];
C = cross(A, B);
assert(isequal(size(C), [2 3]), 'cross 2x3 auto-dim size');
assert(isequal(C(1,:), cross([1 0 0], [0 1 0])), 'cross 2x3 row 1');
assert(isequal(C(2,:), cross([0 1 0], [0 0 1])), 'cross 2x3 row 2');

% 3x3 matrix: first dimension of size 3 is dim 1
A2 = [1 0 0; 0 1 0; 0 0 1];
B2 = [0 0 1; 1 0 0; 0 1 0];
C2 = cross(A2, B2);
assert(isequal(size(C2), [3 3]), 'cross 3x3 auto-dim size');
assert(isequal(C2(:,1), cross([1;0;0], [0;1;0])), 'cross 3x3 col 1');
assert(isequal(C2(:,2), cross([0;1;0], [0;0;1])), 'cross 3x3 col 2');
assert(isequal(C2(:,3), cross([0;0;1], [1;0;0])), 'cross 3x3 col 3');

% --- cross(A,B,dim) with explicit dim ---

% 3x2 matrix, cross along dim 1 (size=3)
A3 = [1 4; 2 5; 3 6];
B3 = [7 10; 8 11; 9 12];
C3 = cross(A3, B3, 1);
assert(isequal(size(C3), [3 2]), 'cross dim=1 size');
assert(isequal(C3(:,1), cross([1;2;3], [7;8;9])), 'cross dim=1 col 1');
assert(isequal(C3(:,2), cross([4;5;6], [10;11;12])), 'cross dim=1 col 2');

% 2x3 matrix, cross along dim 2 (size=3)
A4 = [1 2 3; 4 5 6];
B4 = [7 8 9; 10 11 12];
C4 = cross(A4, B4, 2);
assert(isequal(size(C4), [2 3]), 'cross dim=2 size');
assert(isequal(C4(1,:), cross([1 2 3], [7 8 9])), 'cross dim=2 row 1');
assert(isequal(C4(2,:), cross([4 5 6], [10 11 12])), 'cross dim=2 row 2');

% 3x3 matrix with explicit dim=2 (override auto-detect which would pick dim 1)
A5 = [1 2 3; 4 5 6; 7 8 9];
B5 = [10 11 12; 13 14 15; 16 17 18];
C5 = cross(A5, B5, 2);
assert(isequal(size(C5), [3 3]), 'cross 3x3 dim=2 size');
assert(isequal(C5(1,:), cross([1 2 3], [10 11 12])), 'cross 3x3 dim=2 row 1');
assert(isequal(C5(2,:), cross([4 5 6], [13 14 15])), 'cross 3x3 dim=2 row 2');
assert(isequal(C5(3,:), cross([7 8 9], [16 17 18])), 'cross 3x3 dim=2 row 3');

% --- 3-D arrays ---

% 3x2x2 array, cross along dim 1 (auto-detected, size=3)
A6 = reshape(1:12, [3 2 2]);
B6 = reshape(13:24, [3 2 2]);
C6 = cross(A6, B6);
assert(isequal(size(C6), [3 2 2]), 'cross 3D auto-dim size');
% Check each "column" along dim 1
assert(isequal(C6(:,1,1), cross(A6(:,1,1), B6(:,1,1))), 'cross 3D (:,1,1)');
assert(isequal(C6(:,2,1), cross(A6(:,2,1), B6(:,2,1))), 'cross 3D (:,2,1)');
assert(isequal(C6(:,1,2), cross(A6(:,1,2), B6(:,1,2))), 'cross 3D (:,1,2)');
assert(isequal(C6(:,2,2), cross(A6(:,2,2), B6(:,2,2))), 'cross 3D (:,2,2)');

% 2x3x2 array, cross along dim 2 (auto or explicit)
A7 = reshape(1:12, [2 3 2]);
B7 = reshape(13:24, [2 3 2]);
C7 = cross(A7, B7, 2);
assert(isequal(size(C7), [2 3 2]), 'cross 3D dim=2 size');
assert(isequal(C7(1,:,1), cross(A7(1,:,1), B7(1,:,1))), 'cross 3D dim=2 (1,:,1)');
assert(isequal(C7(2,:,1), cross(A7(2,:,1), B7(2,:,1))), 'cross 3D dim=2 (2,:,1)');
assert(isequal(C7(1,:,2), cross(A7(1,:,2), B7(1,:,2))), 'cross 3D dim=2 (1,:,2)');
assert(isequal(C7(2,:,2), cross(A7(2,:,2), B7(2,:,2))), 'cross 3D dim=2 (2,:,2)');

% 2x4x3 array, cross along dim 3 (explicit)
A8 = reshape(1:24, [2 4 3]);
B8 = reshape(25:48, [2 4 3]);
C8 = cross(A8, B8, 3);
assert(isequal(size(C8), [2 4 3]), 'cross 3D dim=3 size');
assert(isequal(reshape(C8(1,1,:), [1 3]), cross(reshape(A8(1,1,:), [1 3]), reshape(B8(1,1,:), [1 3]))), 'cross 3D dim=3 (1,1,:)');
assert(isequal(reshape(C8(2,3,:), [1 3]), cross(reshape(A8(2,3,:), [1 3]), reshape(B8(2,3,:), [1 3]))), 'cross 3D dim=3 (2,3,:)');

disp('SUCCESS');
