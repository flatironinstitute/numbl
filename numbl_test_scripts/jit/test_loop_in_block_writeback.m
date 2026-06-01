% A loop nested in an if / switch / try must write back the variables it
% assigns, even ones first read AFTER the enclosing block. Regression: the
% loop's live-out analysis only scanned its immediate sibling list, so a
% loop-internal variable consumed after the enclosing block was dropped
% when the loop JIT-compiled as its own unit.
%
% The unsuppressed assignment below keeps the script on the interpreter, so
% each inner loop is dispatched (and JIT-compiled) as its own unit — the
% path where the writeback was dropped.

n = 5

% loop inside if
if n > 0
  for k = 1:n
    s = k * 10;
  end
end
assert(s == 50);

% loop inside switch
switch n
  case 5
    for k = 1:3
      sq = k * k;
    end
end
assert(sq == 9);

% loop inside try
try
  for k = 1:4
    last = k + 100;
  end
catch
end
assert(last == 104);

disp('SUCCESS')
