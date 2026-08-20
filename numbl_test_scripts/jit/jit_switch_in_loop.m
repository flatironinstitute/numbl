% `switch` is lowered to an if/elseif chain, so a function containing one no
% longer forces its caller onto the interpreter.
%
% Before this, `statement type 'Switch' not supported` declined every unit
% that reached a switch — and because a call chain is compiled as a whole,
% one switch deep inside a helper stopped the entire chain from compiling.
%
% Text subjects compare with strcmp, numeric ones with ==, and a
% `case {a, b}` list is the OR of the two tests.

% Text subject, including a cell case list and the otherwise arm.
total = 0;
for k = 1:20
    %!numbl:assert_jit
    total = total + pick('a', k) + pick('b', k) + pick('c', k) + pick('z', k);
end
assert(total == 44080, 'text switch');

% Numeric subject, and a subject that is an expression (evaluated once).
tot2 = 0;
for k = 1:12
    %!numbl:assert_jit
    tot2 = tot2 + byval(k) + bymod(k);
end
assert(tot2 == 572, 'numeric switch');

% A switch whose subject is a string literal, matched against a char case.
assert(pick("a", 1) == 10, 'string subject vs char case');

% Shadowing: a local named like the subject must still read the variable.
kind = 'b';
assert(pick(kind, 2) == 200, 'variable subject');

disp('SUCCESS')

function y = pick(kind, k)
switch kind
    case 'a'
        y = 10 * k;
    case {'b', 'c'}
        y = 100 * k;
    otherwise
        y = -1;
end
end

function y = byval(k)
switch k
    case 1
        y = 5;
    case {2, 3}
        y = 6;
    otherwise
        y = 7;
end
end

function y = bymod(k)
% The subject is an expression: it must be evaluated once, then compared.
switch mod(k, 3)
    case 0
        y = 100;
    case 1
        y = 20;
    otherwise
        y = 3;
end
end
