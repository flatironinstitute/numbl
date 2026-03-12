% Test variable argument functions

% nargin default value
assert(myadd(5) == 15);
assert(myadd(5, 3) == 8);

% varargin
assert(mysum(1, 2, 3) == 6);
assert(mysum(10, 20) == 30);
assert(mysum(7) == 7);

% varargin forwarding with cell expansion
a = {1, 2, 3};
forward_varargs(a{:})

disp('SUCCESS')

function forward_varargs(varargin)
    check_forwarded(8, varargin{:})
end

function check_forwarded(a, b, c, d)
    assert(a == 8)
    assert(b == 1)
    assert(c == 2)
    assert(d == 3)
end

function result = myadd(a, b)
  if nargin < 2
    b = 10;
  end
  result = a + b;
end

function result = mysum(varargin)
  result = 0;
  for i = 1:nargin
    result = result + varargin{i};
  end
end
