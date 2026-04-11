% Scalar tensor indexed assignment inside loops — exercises the loop-JIT
% path for `t(i) = v` / `t(i,j) = v` / `t(i,j,k) = v`. If the JIT writes
% to the wrong slot, the assertions below will catch it.

% 1D scalar write
t = zeros(100, 1);
for i = 1:100
    t(i) = i * 2;
end
assert(t(1) == 2, '1D: first write failed');
assert(t(50) == 100, '1D: middle write failed');
assert(t(100) == 200, '1D: last write failed');

% 1D scalar write with computed index
a = zeros(50, 1);
for i = 1:25
    a(2*i - 1) = i;
    a(2*i) = -i;
end
s = 0;
for i = 1:50
    s = s + a(i);
end
assert(s == 0, '1D: interleaved +/- sum should be zero');

% 2D scalar write (column-major)
m = zeros(5, 4);
for i = 1:5
    for j = 1:4
        m(i, j) = i * 10 + j;
    end
end
assert(m(1, 1) == 11, '2D: (1,1) failed');
assert(m(5, 4) == 54, '2D: (5,4) failed');
assert(m(3, 2) == 32, '2D: (3,2) failed');

% 2D write with condition — mimics the ptloop pattern. Compute a
% reference sum in-loop without any indexed assign, then replay through
% the assigned buffers and verify they agree.
npts = 100;
nrect = 30;
hits_i = zeros(npts * nrect, 1);
hits_j = zeros(npts * nrect, 1);
nhit = 0;
nhit_ref = 0;
sum_i_ref = 0;
sum_j_ref = 0;
for i = 1:npts
    for j = 1:nrect
        if mod(i + j, 7) == 0
            nhit = nhit + 1;
            hits_i(nhit) = i;
            hits_j(nhit) = j;
            nhit_ref = nhit_ref + 1;
            sum_i_ref = sum_i_ref + i;
            sum_j_ref = sum_j_ref + j;
        end
    end
end
sum_i = 0;
sum_j = 0;
for k = 1:nhit
    sum_i = sum_i + hits_i(k);
    sum_j = sum_j + hits_j(k);
end
assert(nhit == nhit_ref, 'ptloop: nhit mismatch');
assert(sum_i == sum_i_ref, 'ptloop: sum_i mismatch');
assert(sum_j == sum_j_ref, 'ptloop: sum_j mismatch');

% While loop with scalar write
w = zeros(20, 1);
k = 0;
while k < 20
    k = k + 1;
    w(k) = k * k;
end
assert(w(1) == 1, 'while: first write failed');
assert(w(20) == 400, 'while: last write failed');

% 3D scalar write
t3 = zeros(3, 4, 5);
for k = 1:5
    for j = 1:4
        for i = 1:3
            t3(i, j, k) = i + 10*j + 100*k;
        end
    end
end
assert(t3(1, 1, 1) == 111, '3D: (1,1,1) failed');
assert(t3(3, 4, 5) == 543, '3D: (3,4,5) failed');
assert(t3(2, 3, 4) == 432, '3D: (2,3,4) failed');

disp('SUCCESS');
