% Test functions calling functions calling functions across files
% chain_outer -> chain_middle -> chain_inner

% chain_inner(x) = x^2 + 1
% chain_middle(x) = chain_inner(x) + chain_inner(x+1) = (x^2+1) + ((x+1)^2+1)
% chain_outer(x) = chain_middle(x) * 2

% Direct call to inner
assert(chain_inner(3) == 10, '3^2 + 1 = 10');
assert(chain_inner(0) == 1, '0^2 + 1 = 1');

% Call to middle (calls inner twice)
% chain_middle(3) = (9+1) + (16+1) = 10 + 17 = 27
assert(chain_middle(3) == 27);

% Call to outer (calls middle, which calls inner)
% chain_outer(3) = 27 * 2 = 54
assert(chain_outer(3) == 54);

% Test with different values
assert(chain_outer(0) == 6, '((0+1)+(1+1))*2 = (1+2)*2 = 6');
assert(chain_outer(1) == 14, '((1+1)+(4+1))*2 = (2+5)*2 = 14');

% Nested call results used in expressions
r = chain_outer(2) + chain_inner(2);
% __inferred_type_str(r) would be "Number" with specialization enabled
% chain_outer(2) = ((4+1)+(9+1))*2 = (5+10)*2 = 30
% chain_inner(2) = 5
assert(r == 35);

% Local function calling file function
assert(local_caller(4) == 34);

disp('SUCCESS')

function r = local_caller(x)
    % Local function calls a file-level function
    r = chain_inner(x) * 2;
end
