% Storing an empty RHS into an empty selection is allowed whatever the
% RHS shape.
A = zeros(0,16,5);
A(:,:,1:0) = zeros(0,0,0);
assert(isequal(size(A), [0 16 5]));
A(:,:,1:0) = zeros(3,0);
assert(isequal(size(A), [0 16 5]));
A(:,:,1:0) = [];
assert(isequal(size(A), [0 16 5]));
B = A(:,:,1:0);
assert(isequal(size(B), [0 16 0]));
D = [];
E = D(:,:,1:0);
assert(isequal(size(E), [0 0 0]));

F = zeros(2,3);
F(:,1:0) = zeros(0,0);
assert(isequal(size(F), [2 3]));
F(1:0,:) = zeros(5,0);
assert(isequal(size(F), [2 3]));
H = zeros(2,3);
H(1:0) = zeros(4,0);
assert(isequal(size(H), [2 3]));

c = cell(2,3);
c(:,1:0) = cell(0,0);
assert(isequal(size(c), [2 3]));
s = 'abc';
s(1:0) = '';
assert(isequal(size(s), [1 3]));

K = zeros(2,3);
K(:,1:0) = 5;
assert(isequal(size(K), [2 3]));
L = zeros(2,3);
threw = false;
try
  L(:,1:0) = [1 2];
catch
  threw = true;
end
assert(threw);

% A classdef property setter that copies an empty N-d slice back.
obj = empty_region_holder(4);
obj = obj.resize(8);
assert(isequal(size(obj.datastor), [0 16 8]));
disp('SUCCESS');
