% Test isstrprop builtin

% Char vector — alpha
TF = isstrprop('123 Maple Street', 'alpha');
expected = logical([0 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1]);
assert(isequal(TF, expected), 'alpha char vec');

% Char vector — digit
TF = isstrprop('123 Maple Street', 'digit');
expected = logical([1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0]);
assert(isequal(TF, expected), 'digit char vec');

% String scalar — digit
TF = isstrprop("123 Maple Street", 'digit');
assert(isequal(TF, expected), 'digit string scalar');

% Whitespace
TF = isstrprop('a b  c', 'wspace');
assert(isequal(TF, logical([0 1 0 1 1 0])), 'wspace');

% Upper / lower
assert(isequal(isstrprop('aBcD', 'lower'), logical([1 0 1 0])), 'lower');
assert(isequal(isstrprop('aBcD', 'upper'), logical([0 1 0 1])), 'upper');

% Punctuation
TF = isstrprop('Hello!', 'punct');
assert(isequal(TF, logical([0 0 0 0 0 1])), 'punct');

% Hex digits
TF = isstrprop('0xFAce!', 'xdigit');
assert(isequal(TF, logical([1 0 1 1 1 1 0])), 'xdigit');

% alphanum
TF = isstrprop('a1!', 'alphanum');
assert(isequal(TF, logical([1 1 0])), 'alphanum');

% print: graphic + space
TF = isstrprop(['a' ' ' char(0)], 'print');
assert(isequal(TF, logical([1 1 0])), 'print');

% cntrl
TF = isstrprop([char(0) 'a' char(9)], 'cntrl');
assert(isequal(TF, logical([1 0 1])), 'cntrl');

% Numeric input — treated as Unicode code points
X = [77 65 84 76 65 66];
TF = isstrprop(X, 'alpha');
assert(isequal(TF, logical([1 1 1 1 1 1])), 'numeric alpha');

% ForceCellOutput on a char vector
TF = isstrprop('Hi!', 'punct', 'ForceCellOutput', true);
assert(iscell(TF), 'forcecell returns cell');
assert(numel(TF) == 1, 'forcecell single entry');
assert(isequal(TF{1}, logical([0 0 1])), 'forcecell content');

% Cell input → cell output
C = {'abc'; '12'};
TF = isstrprop(C, 'alpha');
assert(iscell(TF), 'cell input → cell output');
assert(isequal(TF{1}, logical([1 1 1])), 'cell entry 1');
assert(isequal(TF{2}, logical([0 0])), 'cell entry 2');

disp('SUCCESS');
