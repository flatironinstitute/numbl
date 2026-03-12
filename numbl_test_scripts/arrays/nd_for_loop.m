% 3D tensor: for loop iteration

% In MATLAB, for x = A iterates over columns.
% For a 3D array [2,3,2], it iterates 3*2=6 times,
% each x is a [2,1] column vector (all columns across pages).

a = reshape(1:12, 2, 3, 2);
count = 0;
s = 0;
for x = a
    count = count + 1;
    s = s + x(1);
end

% Should iterate 6 times (3 cols * 2 pages)
assert(count == 6)
% Sum of first elements of each column:
% Page 1 cols: 1, 3, 5; Page 2 cols: 7, 9, 11
% Total = 1+3+5+7+9+11 = 36
assert(s == 36)

% ── Simpler case: [2,2,2] ────────────────────────────────────────
b = reshape(1:8, 2, 2, 2);
vals = zeros(1, 4);
idx = 1;
for col = b
    vals(idx) = col(1);
    idx = idx + 1;
end
% 4 columns: [1;2], [3;4], [5;6], [7;8]
assert(vals(1) == 1)
assert(vals(2) == 3)
assert(vals(3) == 5)
assert(vals(4) == 7)

disp('SUCCESS')
