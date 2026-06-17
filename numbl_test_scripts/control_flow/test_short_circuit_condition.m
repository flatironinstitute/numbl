% Test that the element-wise `|` and `&` operators short-circuit when used
% at the top of an `if`/`while`/`elseif` condition with a scalar left
% operand. This matches MATLAB, where e.g. `if nargin<4 | isempty(x)` must
% NOT evaluate `isempty(x)` when `nargin<4` (so `x` may be undefined).
% This is the classic DistMesh `if nargin<N | isempty(...)` idiom.

% `|` short-circuits: left scalar true -> right (referencing an undefined
% variable) is never evaluated.
result = scfun_or();
assert(result == 1, '| should short-circuit and never touch undefinedvar');

% `&` short-circuits: left scalar false -> right is never evaluated.
result = scfun_and();
assert(result == 1, '& should short-circuit and never touch undefinedvar');

% elseif conditions short-circuit too.
result = scfun_elseif();
assert(result == 2, 'elseif | should short-circuit');

% while conditions short-circuit too. On the first iterations `k < 3` is
% true, so the OR short-circuits and `defined_later` (undefined at first)
% is never evaluated. The body defines it before `k < 3` becomes false, so
% the final condition check can evaluate the right side without error.
k = 0;
while k < 3 | defined_later
    k = k + 1;
    defined_later = false;
end
assert(k == 3, 'while | should short-circuit until k reaches 3');

% Non-scalar `|` still evaluates element-wise (no short-circuit).
result = false;
if [1, 0] | [0, 1]
    result = true;
end
assert(result, 'non-scalar | must be element-wise: [1 0]|[0 1] = [1 1]');

result = false;
if [1, 0] | [0, 0]
    result = true;
end
assert(~result, 'non-scalar | element-wise: [1 0]|[0 0] = [1 0] is not all-true');

% Non-scalar `&` still element-wise.
result = false;
if [1, 1] & [1, 0]
    result = true;
end
assert(~result, 'non-scalar & element-wise: [1 1]&[1 0] = [1 0] is not all-true');

% Scalar OR where left is false: result follows the right operand.
if false | true
    result = true;
else
    result = false;
end
assert(result, 'false | true should be true');

% Scalar AND where left is true: result follows the right operand.
if true & false
    result = true;
else
    result = false;
end
assert(~result, 'true & false should be false');

disp('SUCCESS');

function r = scfun_or()
    if nargin < 1 | isempty(undefinedvar)
        r = 1;
    else
        r = 2;
    end
end

function r = scfun_and()
    if nargin > 0 & isempty(undefinedvar)
        r = 2;
    else
        r = 1;
    end
end

function r = scfun_elseif()
    x = 0;
    if x == 1
        r = 0;
    elseif x == 0 | isempty(undefinedvar)
        r = 2;
    else
        r = 3;
    end
end
