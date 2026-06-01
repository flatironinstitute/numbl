% DIAGNOSIS: Defining an anonymous function at top level makes the whole-scope
% JS-JIT run all side effects, then bail when marshaling the handle-typed output
% variable back to the interpreter env (jitToNumbl can't convert the {} handle
% struct), so the interpreter RE-RUNS the entire script -> all output duplicated.
% The handle `g` is never even called here.
%
% --opt 0 output:
% 3
%
% --opt 1 output:
% 3
% 3
g = @(x) x*2;
disp(3)
