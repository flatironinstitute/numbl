% Test display formatting for various types (display.ts coverage)

% Logical display
disp(true);
disp(false);

% Char display
disp('hello');

% Complex number display
disp(3+4i);
disp(3-4i);
disp(0+5i);
disp(3+0i);

% Empty tensor
disp([]);

% Scalar tensor
disp([42]);

% Complex scalar tensor
disp([3+4i]);

% 2D tensor
disp([1 2 3; 4 5 6]);

% 3D tensor
A = zeros(2,2,2);
A(1,1,1) = 1; A(2,2,1) = 2; A(1,1,2) = 3; A(2,2,2) = 4;
disp(A);

% Cell display
disp({1, 'hello', 3+4i});

% Struct display
s = struct('name', 'test', 'value', 42);
disp(s);

% Function handle display
f = @sin;
disp(f);

% Struct array
sa(1).x = 1;
sa(1).y = 2;
sa(2).x = 3;
sa(2).y = 4;
disp(sa);

disp('SUCCESS');
