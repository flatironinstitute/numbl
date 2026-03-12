% Test type conversion paths (convert.ts coverage)

% char to number
x = double('A');
assert(x == 65);

% logical to double
x = double(true);
assert(x == 1);
x = double(false);
assert(x == 0);

% Complex with zero imaginary to real
z = 5 + 0i;
x = real(z);
assert(x == 5);

% Boolean logic on tensors
if [1 2 3]
    x = 1;
else
    x = 0;
end
assert(x == 1);

% Boolean logic on tensor with zero
if [1 0 3]
    x = 1;
else
    x = 0;
end
assert(x == 0);

% Boolean logic on complex tensor
if [1+1i 2+2i]
    x = 1;
else
    x = 0;
end
assert(x == 1);

% Boolean logic on char
if 'a'
    x = 1;
else
    x = 0;
end
assert(x == 1);

% Boolean logic on complex number
if 0+1i
    x = 1;
else
    x = 0;
end
assert(x == 1);

% num2str
assert(strcmp(num2str(42), '42'));

disp('SUCCESS');
