% Test cumulative functions (cumprod, cummax, cummin) and diff

%% cumprod - cumulative product
% Row vector
cp = cumprod([1, 2, 3, 4]);
assert(cp(1) == 1);
assert(cp(2) == 2);
assert(cp(3) == 6);
assert(cp(4) == 24);

% Column vector
cp2 = cumprod([2; 3; 5]);
assert(cp2(1) == 2);
assert(cp2(2) == 6);
assert(cp2(3) == 30);

% Matrix (along dim 1 = column-wise by default)
M = [1, 2; 3, 4; 5, 6];
cp3 = cumprod(M);
% Column 1: 1, 3, 15; Column 2: 2, 8, 48
assert(cp3(1,1) == 1);
assert(cp3(2,1) == 3);
assert(cp3(3,1) == 15);
assert(cp3(1,2) == 2);
assert(cp3(2,2) == 8);
assert(cp3(3,2) == 48);

% Scalar
assert(cumprod(5) == 5);

%% cummax - cumulative maximum
% Row vector
cm = cummax([3, 1, 4, 1, 5, 2]);
assert(cm(1) == 3);
assert(cm(2) == 3);
assert(cm(3) == 4);
assert(cm(4) == 4);
assert(cm(5) == 5);
assert(cm(6) == 5);

% Column vector
cm2 = cummax([1; 5; 3; 7; 2]);
assert(cm2(1) == 1);
assert(cm2(2) == 5);
assert(cm2(3) == 5);
assert(cm2(4) == 7);
assert(cm2(5) == 7);

% Matrix (along dim 1 by default)
M2 = [3, 1; 1, 4; 5, 2];
cm3 = cummax(M2);
assert(cm3(1,1) == 3);
assert(cm3(2,1) == 3);
assert(cm3(3,1) == 5);
assert(cm3(1,2) == 1);
assert(cm3(2,2) == 4);
assert(cm3(3,2) == 4);

% Scalar
assert(cummax(7) == 7);

%% cummin - cumulative minimum
% Row vector
cn = cummin([3, 1, 4, 1, 5, 2]);
assert(cn(1) == 3);
assert(cn(2) == 1);
assert(cn(3) == 1);
assert(cn(4) == 1);
assert(cn(5) == 1);
assert(cn(6) == 1);

% Column vector
cn2 = cummin([5; 3; 7; 1; 4]);
assert(cn2(1) == 5);
assert(cn2(2) == 3);
assert(cn2(3) == 3);
assert(cn2(4) == 1);
assert(cn2(5) == 1);

% Matrix (along dim 1 by default)
cn3 = cummin(M2);
assert(cn3(1,1) == 3);
assert(cn3(2,1) == 1);
assert(cn3(3,1) == 1);
assert(cn3(1,2) == 1);
assert(cn3(2,2) == 1);
assert(cn3(3,2) == 1);

% Scalar
assert(cummin(7) == 7);

%% diff - differences
% Row vector: diff([1, 3, 6, 10]) = [2, 3, 4]
d = diff([1, 3, 6, 10]);
assert(length(d) == 3);
assert(d(1) == 2);
assert(d(2) == 3);
assert(d(3) == 4);

% Column vector
d2 = diff([2; 5; 11]);
assert(d2(1) == 3);
assert(d2(2) == 6);

% Matrix (along dim 1 by default)
M3 = [1, 4; 2, 8; 5, 16];
d3 = diff(M3);
% Result is 2x2: [1, 4; 3, 8]
assert(size(d3, 1) == 2);
assert(size(d3, 2) == 2);
assert(d3(1,1) == 1);
assert(d3(2,1) == 3);
assert(d3(1,2) == 4);
assert(d3(2,2) == 8);

% Higher-order diff: diff(x, n) applies diff n times
d4 = diff([1, 4, 9, 16], 2);
% diff once: [3, 5, 7], diff twice: [2, 2]
assert(length(d4) == 2);
assert(d4(1) == 2);
assert(d4(2) == 2);

disp('SUCCESS');
