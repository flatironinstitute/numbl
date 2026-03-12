% Test recursive functions

assert(fib(0) == 0);
assert(fib(1) == 1);
assert(fib(5) == 5);
assert(fib(10) == 55);

assert(fact(1) == 1);
assert(fact(5) == 120);
assert(fact(10) == 3628800);

disp('SUCCESS')

function result = fib(n)
  if n <= 1
    result = n;
  else
    result = fib(n-1) + fib(n-2);
  end
end

function result = fact(n)
  if n <= 1
    result = 1;
  else
    result = n * fact(n-1);
  end
end
