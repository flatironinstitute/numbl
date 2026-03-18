% Test that eval inside a function can access the function's local variables

% --- Original bug report: arithmetic via string concatenation ---
function Z = testeval(X, operand, Y)
  eval(['Z = X', operand, 'Y;']);
end

assert(testeval(3, '+', 4) == 7);
assert(testeval(10, '-', 3) == 7);
assert(testeval(5, '*', 6) == 30);

% --- eval reading local variables ---
function r = eval_read(a, b)
  eval('r = a + b;');
end

assert(eval_read(10, 20) == 30);

% --- eval writing to existing local variable ---
function r = eval_overwrite(x)
  r = x;
  eval('r = r * 2;');
end

assert(eval_overwrite(5) == 10);

% --- multiple evals reading/writing same variables ---
function r = multi_eval(a, b)
  r = 0;
  eval('r = a + b;');
  eval('r = r * 2;');
end

assert(multi_eval(3, 4) == 14);

% --- eval inside control flow (if/for) ---
function r = eval_in_loop(n)
  r = 0;
  for i = 1:n
    eval('r = r + i;');
  end
end

assert(eval_in_loop(5) == 15);

function r = eval_in_if(x)
  r = 0;
  if x > 0
    eval('r = x;');
  end
end

assert(eval_in_if(7) == 7);
assert(eval_in_if(-3) == 0);

% --- eval with matrix operations ---
function r = eval_matrix()
  A = [1 2; 3 4];
  eval('r = A(2, 1);');
end

assert(eval_matrix() == 3);

% --- eval at script level (expression evaluation) ---
eval('disp(''eval at script level works'')');

disp('SUCCESS');
