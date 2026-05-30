% DIAGNOSIS: isequal(logical, number) diverges. JIT JS-emit compares a
%   logical scalar (a JS boolean) to a number with strict ===:
%   `true === 1` is false. The interpreter coerces logical->number
%   (numOf) so 1===1 -> true.
%   Root: src/numbl-core/jit/builtins/defs/logical/isequal.ts pairJs
%         (rs===rs branch returns `(aJs === bJs ? 1 : 0)`)
% --opt 0:  s1=10000 (isequal(true,1)=1)   s2=10000 (isequal(false,0)=1)
% --opt 1:  s1=0                            s2=0
function r = g(a,b)
  r = 0;
  for k=1:200
    r = r + isequal(a,b);
  end
end
s1=0; s2=0;
for k=1:50
  s1 = s1 + g(true, 1);
  s2 = s2 + g(false, 0);
end
disp(s1); disp(s2)
