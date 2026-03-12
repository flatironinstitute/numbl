% switch / case / otherwise

x = 2;
switch x
  case 1
    label = 'one';
  case 2
    label = 'two';
  case 3
    label = 'three';
  otherwise
    label = 'other';
end
assert(strcmp(label, 'two'))

% switch with string
color = 'red';
switch color
  case 'red'
    val = 1;
  case 'green'
    val = 2;
  case 'blue'
    val = 3;
  otherwise
    val = 0;
end
assert(val == 1)

% switch with cell array case (matches any)
n = 5;
switch n
  case {1, 2, 3}
    cat = 'small';
  case {4, 5, 6}
    cat = 'medium';
  otherwise
    cat = 'large';
end
assert(strcmp(cat, 'medium'))

% otherwise
z = 99;
switch z
  case 1
    r = 'one';
  otherwise
    r = 'unknown';
end
assert(strcmp(r, 'unknown'))

disp('SUCCESS')
