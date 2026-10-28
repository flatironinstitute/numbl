% DIAGNOSIS: the JITs ignored `arguments` blocks (which the interpreter
% did not enforce either). Once the interpreter validates, a compiled call
% must not skip size/class conversion or validators: at opt1 `scale([1;2;3])`
% returned the column unreshaped and `scale(-1)` / `scale(ones(2))` ran
% instead of raising. Calls are now validated on the interpreter before a
% top-level call is offered to the JIT, and specialization declines
% callees whose blocks need run-time work (the loop in `loop_calls`).
% Also covers the JS-JIT returning a logical parameter unchanged as double
% (`ident(true)` printed `double` at opt1).
function y = scale(x, k)
  arguments
    x (1,:) double {mustBeNonnegative}
    k (1,1) double = 2
  end
  y = x * k;
end
function y = ident(x)
  y = x;
end
function r = pick(x)
  arguments
    x (1,:) double
  end
  r = x(5);
end
function s = loop_calls(n)
  s = 0;
  for i = 1:n
    s = s + sum(scale([i; i]));
  end
end
r = scale(true);
disp(class(r)); disp(r)
disp(scale([1; 2; 3], 3))
disp(loop_calls(50))
disp(class(ident(true)))
disp(class(ident([true false])))
try
  scale(-1);
catch e
  disp(e.identifier); disp(e.message)
end
try
  scale(ones(2));
catch e
  disp(e.identifier); disp(e.message)
end
for i = 1:30
  v = scale(i);
end
disp(v)
for i = 1:3
  try
    pick([1 2]);
  catch e
    disp(e.message); disp(e.stack(1).name)
  end
end
pick(1:3)
