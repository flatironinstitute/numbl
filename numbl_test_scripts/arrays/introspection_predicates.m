% Test shape/type introspection predicates:
% isvector, ismatrix, isrow, iscolumn, isfloat, isinteger

%% isvector - true for 1xN, Nx1, and scalars
assert(isvector([1, 2, 3]) == true);      % row vector
assert(isvector([1; 2; 3]) == true);      % column vector
assert(isvector(5) == true);              % scalar is a vector
assert(isvector([1, 2; 3, 4]) == false);  % matrix is not a vector
assert(isvector(ones(1, 1)) == true);     % 1x1 tensor
assert(isvector('hello') == true);        % char vector

%% isrow - true only for 1xN
assert(isrow([1, 2, 3]) == true);         % row vector
assert(isrow([1; 2; 3]) == false);        % column vector is not a row
assert(isrow(5) == true);                 % scalar is 1x1, so it's a row
assert(isrow([1, 2; 3, 4]) == false);     % matrix

%% iscolumn - true only for Nx1
assert(iscolumn([1; 2; 3]) == true);      % column vector
assert(iscolumn([1, 2, 3]) == false);     % row vector is not a column
assert(iscolumn(5) == true);              % scalar is 1x1, so it's a column
assert(iscolumn([1, 2; 3, 4]) == false);  % matrix

%% ismatrix - true for 2D arrays (including scalars and vectors)
assert(ismatrix([1, 2; 3, 4]) == true);   % 2D matrix
assert(ismatrix([1, 2, 3]) == true);      % row vector is 2D
assert(ismatrix([1; 2; 3]) == true);      % column vector is 2D
assert(ismatrix(5) == true);              % scalar is 2D
assert(ismatrix('abc') == true);          % char array is 2D
% 3D array is not a matrix
A = ones(2, 3, 4);
assert(ismatrix(A) == false);

%% isfloat - true for double/single (numbl uses double)
assert(isfloat(3.14) == true);            % scalar double
assert(isfloat([1, 2, 3]) == true);       % double array
assert(isfloat(true) == false);           % logical is not float
assert(isfloat('abc') == false);          % char is not float

%% isinteger - false for doubles (numbl has no int types)
assert(isinteger(5) == false);            % double, even if whole number
assert(isinteger([1, 2, 3]) == false);    % double array
assert(isinteger(true) == false);         % logical
assert(isinteger('a') == false);          % char

disp('SUCCESS');
