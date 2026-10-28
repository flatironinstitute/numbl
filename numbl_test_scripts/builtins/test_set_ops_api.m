% Set operations, sort, find, unique and friends: complex ordering (abs,
% then angle), char operands, option validation and arity errors.

% --- complex values sort by abs, then angle ---
assert(isequal(sort([1i -1 1 -1i]), [-1i 1 1i -1]));
[s, i] = sort([1i -1 1 -1i 2 1+1i], 'descend');
assert(isequal(s, [2 1+1i -1 1i 1 -1i]) && isequal(i, [5 6 2 1 3 4]));
[s, i] = sort([1i NaN -1 NaN+1i]);
assert(isequal(i, [1 3 2 4]) && isequal(s(1:2), [1i -1]));
[s, i] = sort([2i 1i; -1 3]);
assert(isequal(s, [-1 1i; 2i 3]) && isequal(i, [2 1; 1 2]));
[s, i] = sort([2i 1i; -1 3], 2);
assert(isequal(s, [1i 2i; -1 3]) && isequal(i, [2 1; 1 2]));

% --- setdiff / intersect / union keep the imaginary part ---
assert(isequal(setdiff([1+2i 0 3i 1+2i -1], [3i 1-1i 0 2 3i]), [-1, 1+2i]));
[c, ia] = setdiff([1+2i 0 3i 1+2i -1], [3i 1-1i 0 2 3i]);
assert(isequal(c, [-1, 1+2i]) && isequal(ia, [5; 1]));
assert(isequal(setdiff([1+2i 0 3i 1+2i -1], [3i 1-1i 0 2 3i], 'stable'), [1+2i, -1]));
[c, ia, ib] = intersect([1+2i 3 -1i 2 1-2i], [-1i 1+2i 5 2]);
assert(isequal(c, [-1i 2 1+2i]) && isequal(ia, [3; 4; 1]) && isequal(ib, [1; 4; 2]));
[c, ia, ib] = union([1+2i 3 -2], [-1 2i 1-2i]);
assert(isequal(c, [-1 2i -2 1-2i 1+2i 3]) && isequal(ia, [3; 1; 2]) && isequal(ib, [1; 2; 3]));
assert(isequal(union([1 2], 1i), [1 1i 2]));
[c, ia] = setdiff([1+1i 2; 3 4], [3 4], 'rows');
assert(isequal(c, [1+1i 2]) && ia == 1);

% --- unique sorts complex by abs, then angle ---
[c, ia, ic] = unique([3 1+1i -1 1i 1-1i 2 1i]);
assert(isequal(c, [1i -1 1-1i 1+1i 2 3]));
assert(isequal(ia, [4; 3; 5; 2; 6; 1]) && isequal(ic, [6; 4; 2; 1; 3; 5; 1]));
[c, ia, ic] = unique([3 1+1i -1 1i 1-1i 2 1i], 'stable');
assert(isequal(c, [3 1+1i -1 1i 1-1i 2]) && isequal(ic(end), 4));
c = unique([1i NaN 2 NaN+1i -2]);
assert(isequal(c(1:3), [1i 2 -2]) && all(isnan(c(4:5))) && imag(c(5)) == 1);
assert(isequal(unique(1+1i), 1+1i));

% --- unique 'first' / 'last' ---
[c, ia, ic] = unique([3 1 3 2], 'last');
assert(isequal(c, [1 2 3]) && isequal(ia, [2; 4; 3]) && isequal(ic, [3; 1; 3; 2]));
[~, ia] = unique([3 1 3 2], 'first');
assert(isequal(ia, [2; 4; 1]));

% --- sortrows and groupcounts on complex input ---
[b, i] = sortrows([1+1i 2; 1 3; -1 0; 1i 5]);
assert(isequal(b, [1 3; 1i 5; -1 0; 1+1i 2]) && isequal(i, [2; 4; 3; 1]));
[b, i] = sortrows([1+1i 2; 1 3; -1 0; 1i 5], -1);
assert(isequal(b, [1+1i 2; -1 0; 1i 5; 1 3]) && isequal(i, [1; 3; 4; 2]));
[b, i] = sortrows([1 2i; 1 -2; 1 2]);
assert(isequal(b, [1 2; 1 2i; 1 -2]) && isequal(i, [3; 1; 2]));
[gc, gr, gp] = groupcounts([1+1i; 2; 1+1i; -2]);
assert(isequal(gc, [2; 1; 1]) && isequal(gr, [1+1i; 2; -2]) && isequal(gp, [50; 25; 25]));

% --- sortrows directions and NaN placement ---
[b, i] = sortrows([3 1; 1 2; 3 0], 1, 'descend');
assert(isequal(b, [3 1; 3 0; 1 2]) && isequal(i, [1; 3; 2]));
b = sortrows([3 1; 1 2; 3 0], [1 2], {'descend', 'ascend'});
assert(isequal(b, [3 0; 3 1; 1 2]));
assert(isequal(sortrows([3 1; 1 2; 3 0], 'ascend'), [1 2; 3 0; 3 1]));
[b, i] = sortrows([3 1; 1 2; 3 0], []);
assert(isequal(b, [3 1; 1 2; 3 0]) && isequal(i, [1; 2; 3]));
[~, i] = sortrows([NaN 1; 2 3; 1 2]);
assert(isequal(i, [3; 2; 1]));
[~, i] = sortrows([NaN 1; 2 3; 1 2], -1);
assert(isequal(i, [1; 2; 3]));
[b, i] = sortrows(['bc'; 'ab'; 'ba']);
assert(isequal(b, ['ab'; 'ba'; 'bc']) && ischar(b) && isequal(i, [2; 3; 1]));

% --- ismember compares complex values exactly; char vs char by element ---
[tf, loc] = ismember([1+1i 2 1-1i], [1+1i 2]);
assert(isequal(tf, [true true false]) && isequal(loc, [1 2 0]));
assert(~ismember(1-1i, [1+1i 2]));
assert(~ismember(1i, 1));
[tf, loc] = ismember('b', 'abc');
assert(tf && loc == 2);
[tf, loc] = ismember('bz', 'abc');
assert(isequal(tf, [true false]) && isequal(loc, [2 0]));
assert(ismember('abc', {'abc', 'x'}));
assert(~ismember(NaN, [1 NaN]));
[tf, loc] = ismember([1 2; 1 2; 3 4], [3 4; 1 2; 1 2], 'rows');
assert(isequal(tf, [true; true; true]) && isequal(loc, [2; 2; 1]));
assert(ismember([1+1i 2], [1+1i 2; 0 0], 'rows'));

% --- NaN is never equal to NaN in set operations ---
assert(isequal(size(union(NaN, NaN)), [1 2]));
assert(isempty(intersect(NaN, NaN)));
c = setdiff([NaN 1], NaN);
assert(c(1) == 1 && isnan(c(2)));

% --- char operands ---
assert(isequal(setdiff('abc', 'b'), 'ac'));
assert(isequal(union('abc', 'cd'), 'abcd'));
assert(isequal(intersect('abc', 'cbx'), 'bc'));
assert(isequal(setdiff('cab', 'x', 'stable'), 'cab'));
assert(isequal(setdiff([3 1 2], 'a'), [1 2 3]));
assert(isequal(union([97], 'b'), 'ab'));

% --- find, nnz, nonzeros ---
[r, c, v] = find(1+2i);
assert(r == 1 && c == 1 && isequal(v, 1+2i));
[r, c, v] = find([0 1+1i 2]);
assert(isequal(r, [1 1]) && isequal(c, [2 3]) && isequal(v, [1+1i 2]));
assert(isequal(find('a b'), [1 2 3]));
assert(isequal(nonzeros(1+2i), 1+2i));
assert(isequal(nonzeros('ab'), ['a'; 'b']));
assert(islogical(nonzeros([true false true])));
assert(nnz('abc') == 3);
assert(nnz(sparse([1 0 1i])) == 2);

% --- sort of char ---
[s, i] = sort('hello');
assert(isequal(s, 'ehllo') && ischar(s) && isequal(i, [2 1 3 4 5]));
[s, i] = sort('hello', 'descend');
assert(isequal(s, 'ollhe') && isequal(i, [5 3 4 1 2]));
assert(isequal(sort(['ba'; 'dc'], 2), ['ab'; 'cd']));
assert(isequal(sort([3 1 2], 'Descend'), [3 2 1]));
assert(isequal(sort([3 1 2], "descend"), [3 2 1]));

% --- option and type errors ---
check_err(@() sort([3 1 2], 'foo'), 'MATLAB:sort:sortDirection', 'Sorting direction must be ''ascend'' or ''descend''.');
check_err(@() sort([3 1 2], 1.5), 'MATLAB:sort:notPosInt', 'Dimension argument must be a positive integer.');
check_err(@() sort([3 1 2], 0), 'MATLAB:sort:notPosInt', 'Dimension argument must be a positive integer.');
check_err(@() sort({1}), 'MATLAB:sort:InputType', 'Input argument must be a cell array of character vectors.');
check_err(@() unique([3 1 2], 'foo'), 'MATLAB:UNIQUE:UnknownFlag', 'Unrecognized option "foo". Valid options are "rows", "first", "last", "stable", "sorted", "legacy".');
check_err(@() unique({1, 2}), 'MATLAB:UNIQUE:InputClass', 'Cell array input must be a cell array of character vectors.');
check_err(@() setdiff([3 1 2], 1, 'foo'), 'MATLAB:SETDIFF:UnknownFlag', 'Unrecognized flag ''foo''. Valid flags are ''rows'', ''stable'', ''sorted'', ''legacy''.');
check_err(@() union([3 1 2], 1, 'foo'), 'MATLAB:UNION:UnknownFlag', 'Unrecognized flag ''foo''. Valid flags are ''rows'', ''stable'', ''sorted'', ''legacy''.');
check_err(@() intersect([1 2 3], 2, 'rows'), 'MATLAB:INTERSECT:AandBColnumAgree', 'Inputs A and B must have the same number of columns in the ''rows'' case.');
check_err(@() setdiff({1}, {2}), 'MATLAB:SETDIFF:InputClass', 'Input A of class cell and input B of class cell must be cell arrays of character vectors, unless one is a character vector.');
check_err(@() ismember(1, {'a'}), 'MATLAB:ISMEMBER:InputClass', 'Input A of class double and input B of class cell must be cell arrays of character vectors, unless one is a character vector.');
check_err(@() ismember([1 2], [1 2], 'foo'), 'MATLAB:ISMEMBER:UnknownFlag', 'Unrecognized flag ''foo''. Valid flags are ''rows'', ''legacy''.');
check_err(@() ismember([1 2], [1 2 3], 'rows'), 'MATLAB:ISMEMBER:AandBColnumAgree', 'Inputs A and B must have the same number of columns in the ''rows'' case.');
check_err(@() find([1 0 1], 1.5), 'MATLAB:find:NotScalarInt', 'Second argument must be a positive scalar integer.');
check_err(@() find([1 0 1], -1), 'MATLAB:find:NotScalarInt', 'Second argument must be a positive scalar integer.');
check_err(@() find([1 0 1], 1, 'foo'), 'MATLAB:find:InvalidOption', 'Invalid search option. Must be ''first'' or ''last''.');
check_err(@() find({1}), 'MATLAB:UndefinedFunction', 'Undefined function ''find'' for input arguments of type ''cell''.');
check_err(@() nnz({1}), 'MATLAB:math:mustBeNumericCharOrLogical', 'Invalid data type. Argument must be numeric, char, or logical.');
check_err(@() sortrows([3 1; 1 2], 1.5), 'MATLAB:sortrows:COLmismatchX', 'Column sorting vector must contain integers with absolute value between 1 and the number of columns in the first argument.');
check_err(@() sortrows([3 1; 1 2], 3), 'MATLAB:sortrows:COLmismatchX', 'Column sorting vector must contain integers with absolute value between 1 and the number of columns in the first argument.');
check_err(@() sortrows([3; 1; 2], 'foo'), 'MATLAB:sortrows:DIRnotRecognized', 'Sorting direction must be ''ascend'' or ''descend''.');
check_err(@() uniquetol([1 1+1i]), 'MATLAB:uniquetol:UnsupportedTypeForA', 'First argument must be a real full matrix of type single or double.');
check_err(@() ismembertol(1i, 1), 'MATLAB:ismembertol:UnsupportedTypeForA', 'Input A must be a real full matrix of type single or double.');

% --- arity errors ---
check_err(@() find(), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() find(1, 1, 'first', 1), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() nnz(1, 2), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() ismembc(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() ismember(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() ismember(1, 2, 'rows', 'rows', 'rows'), 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err(@() sort(), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() sortrows(), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() nout3(@sort), 'MATLAB:maxlhs', 'Too many output arguments.');
check_err(@() nout3(@sortrows), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err(@() nout4(@unique), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err(@() nout4(@find), 'MATLAB:maxlhs', 'Too many output arguments.');
assert(nargin(@find) == 3);
assert(nargin(@nnz) == 1);
assert(nargin(@ismember) == 4);

disp('SUCCESS');

function nout3(f)
  [a, b, c] = f(1);
end

function nout4(f)
  [a, b, c, d] = f(1);
end

function check_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('expected %s, got %s', id, e.identifier));
    assert(strcmp(e.message, msg), sprintf('expected "%s", got "%s"', msg, e.message));
    return;
  end
  error('expected error %s', id);
end
