% if / elseif / else

x = 10;
if x > 5
  result = 'big';
else
  result = 'small';
end
assert(strcmp(result, 'big'))

y = 3;
if y > 10
  cat = 'large';
elseif y > 5
  cat = 'medium';
else
  cat = 'small';
end
assert(strcmp(cat, 'small'))

% Nested if
a = 4;
if a > 0
  if a > 2
    sign = 'pos_large';
  else
    sign = 'pos_small';
  end
else
  sign = 'neg';
end
assert(strcmp(sign, 'pos_large'))

disp('SUCCESS')
