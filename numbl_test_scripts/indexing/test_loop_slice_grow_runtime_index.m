% Multi-slot indexed assignment inside a loop where the slot index is a
% runtime expression that grows the array past its initial size. Mirrors
% the pattern in chunkie's `lege.pols`:
%
%     pols(:, 1) = ones(length(xs), 1);
%     pols(:, 2) = xs(:);
%     for k = 1:(n-1)
%         pols(:, k + 2) = pkp1;   % grows `pols` past its initial 2-col size
%     end
%
% The JIT must either auto-grow correctly or bail to the interpreter.
% Silently dropping the writes past the initial size is wrong.

% 1) Grow column index in a loop via `a(:, k+2)` style index
a(:, 1) = 0;
a(:, 2) = 1;
for k = 1:4
    a(:, k + 2) = k * 10;
end
assert(size(a, 1) == 1, '1: row count');
assert(size(a, 2) == 6, '1: col count after grow');
assert(a(1, 1) == 0, '1: col 1');
assert(a(1, 2) == 1, '1: col 2');
assert(a(1, 3) == 10, '1: col 3');
assert(a(1, 4) == 20, '1: col 4');
assert(a(1, 5) == 30, '1: col 5');
assert(a(1, 6) == 40, '1: col 6');

% 2) Same pattern but the array starts as a multi-row column-allocated tensor
b = zeros(3, 2);
b(:, 1) = ones(3, 1);
b(:, 2) = ones(3, 1) * 2;
for k = 1:3
    b(:, k + 2) = ones(3, 1) * (k * 10);
end
assert(size(b, 1) == 3, '2: row count');
assert(size(b, 2) == 5, '2: col count after grow');
assert(b(1, 3) == 10 && b(3, 3) == 10, '2: col 3');
assert(b(2, 5) == 30, '2: col 5');

% 3) Companion ders-style: a second variable assigned the SAME way in the
%    same loop body (this is the exact failure mode in chunkie's lege.pols
%    — both `pols(:, k+2) = ...` and `ders(:, k+2) = ...` are updated).
pols(:, 1) = ones(4, 1);
ders(:, 1) = zeros(4, 1);
pols(:, 2) = ones(4, 1) * 2;
ders(:, 2) = ones(4, 1) * 3;
for k = 1:3
    pols(:, k + 2) = ones(4, 1) * (k + 100);
    ders(:, k + 2) = ones(4, 1) * (k + 200);
end
assert(size(pols, 2) == 5, '3: pols col count');
assert(size(ders, 2) == 5, '3: ders col count');
assert(pols(1, 3) == 101 && pols(4, 5) == 103, '3: pols values');
assert(ders(1, 3) == 201 && ders(4, 5) == 203, '3: ders values');

disp('SUCCESS');
