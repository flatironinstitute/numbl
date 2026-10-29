% Operand checks for the : operator and for & and |.

% : rejects operands that are not numeric, char or logical
bad = {'{1}:3', '1:{3}', 'struct():3', '1:2:{3}', '{1}:2:3'};
for k = 1:numel(bad)
    try
        eval(['y = ' bad{k} ';']);
        error('no error for %s', bad{k});
    catch e
        assert(strcmp(e.identifier, 'MATLAB:colon:inputsMustBeNumericCharLogical'), bad{k});
        assert(strcmp(e.message, 'Inputs must be numeric, char, or logical.'));
    end
end
assert(isequal('a':'c', 'abc'));
assert(isequal(true:3, 1:3));

% & and | reject complex operands
bad = {'complex(1, 1) & 1', '1 | complex(1, 1)', '[complex(1, 1) 2] & [1 1]', ...
       '[1 1] | [2 1i]', 'and(complex(1, 1), 1)', 'or(complex(1, 1), 1)', ...
       'xor(complex(1, 1), 1)'};
for k = 1:numel(bad)
    try
        eval(['y = ' bad{k} ';']);
        error('no error for %s', bad{k});
    catch e
        assert(strcmp(e.identifier, 'MATLAB:math:mustBeReal'), bad{k});
        assert(strcmp(e.message, 'Argument must be real.'));
    end
end
assert(isequal([1 0 2] & [1 1 0], [true false false]));
assert(isequal([1 0 0] | [0 0 3], [true false true]));

disp('SUCCESS');
