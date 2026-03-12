% for loops

% Basic counting loop
total = 0;
for i = 1:5
  total = total + i;
end
assert(total == 15)

% Loop with step
total2 = 0;
for i = 1:2:9
  total2 = total2 + i;
end
assert(total2 == 25)  % 1+3+5+7+9

% Loop over vector
v = [10, 20, 30];
s = 0;
for x = v
  s = s + x;
end
assert(s == 60)

% Nested loops
count = 0;
for i = 1:3
  for j = 1:3
    count = count + 1;
  end
end
assert(count == 9)

% For loop with parentheses (MATLAB allows this)
a = 0;
for (j = 1:3)
  a = a + j;
end
assert(a == 6)

disp('SUCCESS')
