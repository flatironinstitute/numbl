% Test histc builtin

% --- Basic row vector example from MATLAB docs ---
ages = [3,12,24,15,5,74,23,54,31,23,64,75];
binranges = [0,10,25,50,75];
[bincounts, ind] = histc(ages, binranges);
assert(isequal(size(bincounts), [1 5]));
assert(isequal(bincounts, [2 5 1 3 1]));
assert(isequal(ind, [1 2 2 2 1 4 2 4 3 2 4 5]));

% Single-output form
bc = histc(ages, binranges);
assert(isequal(bc, [2 5 1 3 1]));

% --- Column vector input → column output ---
xcol = ages.';
bccol = histc(xcol, binranges);
assert(isequal(size(bccol), [5 1]));
assert(isequal(bccol, [2; 5; 1; 3; 1]));

% --- Values outside the range and at the right edge ---
% bin 1: [0,10), bin 2: [10,25), bin 3: [25,50), bin 4: [50,75), bin 5: scalar 75
x2 = [-1 0 9.999 10 24 25 49 50 75 76 NaN];
[bc2, in2] = histc(x2, binranges);
assert(isequal(bc2, [2 2 2 1 1]));
assert(isequal(in2, [0 1 1 2 2 3 3 4 5 0 0]));

% --- Last bin captures only exact equality with last edge ---
xe = [4 5 5 5];
ber = [0 5];
[bce, ine] = histc(xe, ber);
assert(isequal(bce, [1 3]));
assert(isequal(ine, [1 2 2 2]));

% --- Matrix input, default dim = 1 (operate along columns) ---
M = [1 4; 2 5; 3 6];     % 3x2
br = [0 2 4 6];          % 4 bins (last is exactly 6)
bcm = histc(M, br);
% column 1 = [1;2;3]: bin 1 (1), bin 2 (2,3), bin 3 (none), bin 4 (none) -> [1 2 0 0]
% column 2 = [4;5;6]: bin 1 (none), bin 2 (none), bin 3 (4,5), bin 4 (6) -> [0 0 2 1]
assert(isequal(size(bcm), [4 2]));
assert(isequal(bcm, [1 0; 2 0; 0 2; 0 1]));

% Matrix input with explicit dim = 2 (operate along rows)
N = M.';                 % 2x3 = [1 2 3; 4 5 6]
bcm2 = histc(N, br, 2);
assert(isequal(size(bcm2), [2 4]));
assert(isequal(bcm2, [1 2 0 0; 0 0 2 1]));

% --- ind matches x size for matrices ---
[~, indm] = histc(M, br);
assert(isequal(size(indm), [3 2]));
assert(isequal(indm, [1 3; 2 3; 2 4]));

% --- -inf / inf catch-all bin edges ---
xx = [-100 0 1 5 10 1e9];
edges = [-inf 0 5 inf];
bcx = histc(xx, edges);
% bin1: [-inf,0) -> -100 (1)
% bin2: [0,5)   -> 0,1   (2)
% bin3: [5,inf) -> 5,10,1e9 (3)
% bin4: scalar inf -> none
assert(isequal(bcx, [1 2 3 0]));

disp('SUCCESS');
