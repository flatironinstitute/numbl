% Test operator precedence and associativity edge cases

%% Power associativity (MATLAB: left-associative)
assert(2^3^2 == 64);            % (2^3)^2 = 8^2 = 64, NOT 2^(3^2) = 2^9 = 512
assert(3^2^3 == 729);           % (3^2)^3 = 9^3 = 729, NOT 3^(2^3) = 3^8 = 6561
assert(2^2^2^2 == 256);         % ((2^2)^2)^2 = 4^2^2 = 16^2 = 256

%% Element-wise power associativity (also left-associative)
assert([2].^[3].^[2] == 64);
a = [2 3]; b = [3 2]; c = [2 3];
r = a.^b.^c;
assert(r(1) == 64);             % (2^3)^2 = 64
assert(r(2) == 729);            % (3^2)^3 = 729

%% Power with unary minus in exponent
assert(2^-1 == 0.5);
assert(4^-0.5 == 0.5);
assert(2^-1^2 == 0.25);         % (2^(-1))^2 = 0.5^2 = 0.25

%% Precedence: power vs multiplication
assert(2*3^2 == 18);            % 2*(3^2) = 18, NOT (2*3)^2 = 36
assert(2^3*4 == 32);            % (2^3)*4 = 32
assert(-2^2 == -4);             % -(2^2), unary minus has lower prec than ^

%% Precedence: power vs unary minus
assert((-2)^2 == 4);
assert(-2^2 == -4);
assert((-3)^3 == -27);
assert(-3^3 == -27);            % -(3^3) = -27

%% Precedence: multiplication vs addition
assert(2 + 3 * 4 == 14);
assert(2 * 3 + 4 == 10);
assert(1 + 2 * 3 + 4 == 11);

%% Precedence: division and multiplication (left-to-right)
assert(12 / 3 / 2 == 2);       % (12/3)/2 = 2, NOT 12/(3/2) = 8
assert(12 / 3 * 2 == 8);       % (12/3)*2 = 8
assert(2 * 6 / 3 == 4);        % (2*6)/3 = 4

%% Precedence: addition and subtraction (left-to-right)
assert(10 - 3 - 2 == 5);       % (10-3)-2 = 5, NOT 10-(3-2) = 9
assert(10 - 3 + 2 == 9);
assert(1 + 2 - 3 + 4 - 5 == -1);

%% Precedence: comparison vs arithmetic
assert((3 + 2 > 4) == true);
assert((3 * 2 == 6) == true);
assert((10 - 5 >= 5) == true);
assert((2^3 < 10) == true);

%% Precedence: logical AND vs OR
assert((true | false & false) == true);   % true | (false & false) = true
assert((false & true | true) == true);    % (false & true) | true = true
assert((false | false & true) == false);  % false | (false & true) = false

%% Precedence: NOT vs AND/OR
assert((~false & true) == true);
assert((~true | false) == false);
assert((~(true & false)) == true);

%% Complex mixed expressions
assert(2 + 3 * 4^2 == 50);     % 2 + 3*(4^2) = 2 + 48 = 50
assert((2 + 3) * 4^2 == 80);   % 5 * 16 = 80
assert(2^3 + 4 * 5 - 1 == 27); % 8 + 20 - 1 = 27
assert(abs(2^0.5 - sqrt(2)) < 1e-10);

%% Matrix power associativity
A = [1 1; 0 1];
B = A^2^3;                      % (A^2)^3 = A^6
A6 = A^6;
assert(B(1,1) == A6(1,1));
assert(B(1,2) == A6(1,2));

%% Unary plus/minus chains
assert(--3 == 3);               % -(-3) = 3
assert(---3 == -3);             % -(--3) = -3
assert(+3 == 3);
assert(++3 == 3);
assert(-+3 == -3);
assert(+-3 == -3);

%% Parenthesized power
assert((2^3)^2 == 64);
assert(2^(3^2) == 512);

%% Colon vs arithmetic precedence
v = 1+1:2+3;                   % (1+1):(2+3) = 2:5
assert(length(v) == 4);
assert(v(1) == 2);
assert(v(end) == 5);

v2 = 1:2*3;                    % 1:(2*3) = 1:6
assert(length(v2) == 6);

v3 = 2^2:3^2;                  % (2^2):(3^2) = 4:9
assert(length(v3) == 6);
assert(v3(1) == 4);
assert(v3(end) == 9);

disp('SUCCESS');
