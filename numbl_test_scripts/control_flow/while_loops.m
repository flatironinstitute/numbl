% while loops

n = 1;
while n < 32
  n = n * 2;
end
assert(n == 32)

% while with break
x = 0;
while true
  x = x + 1;
  if x >= 5
    break
  end
end
assert(x == 5)

% while with continue
total = 0;
k = 0;
while k < 10
  k = k + 1;
  if mod(k, 2) == 0
    continue
  end
  total = total + k;
end
assert(total == 25)  % 1+3+5+7+9

disp('SUCCESS')
