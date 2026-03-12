% Test break and continue in loops

% break out of for loop
total = 0;
for i = 1:10
  if i == 5
    break;
  end
  total = total + i;
end
assert(total == 10);  % 1+2+3+4

% continue in for loop
total2 = 0;
for i = 1:10
  if mod(i, 2) == 0
    continue;
  end
  total2 = total2 + i;
end
assert(total2 == 25);  % 1+3+5+7+9

% break out of while loop
n = 0;
while true
  n = n + 1;
  if n >= 7
    break;
  end
end
assert(n == 7);

% nested loops with break
found = 0;
for i = 1:5
  for j = 1:5
    if i * j == 12
      found = i * 10 + j;
      break;
    end
  end
  if found > 0
    break;
  end
end
% 3*4=12, so found = 34
assert(found == 34);

disp('SUCCESS')
