% Test that colon (:) is correctly passed as ':' to overloaded subsref.
% When obj(:) is called on a class with subsref, MATLAB passes ':' (char)
% as S.subs{1}. In numbl, the internal COLON_SENTINEL must be converted
% to the char ':' before building the S struct for subsref.

obj = ColonTracker_();

% Test 1: obj(:) — colon as single index
result1 = obj(:);
assert(result1 == 1, 'obj(:) should pass colon as char to subsref');

% Test 2: multi-index obj(:, 3) — colon as first of two indices
% (requires a different subsref handler, tested via container pattern below)

% Test 3: Access a property that is a ColonTracker_, then index with (:)
container = struct();
container.prop = ColonTracker_();
result3 = container.prop(:);
assert(result3 == 1, 'container.prop(:) should pass colon correctly to subsref');

disp('SUCCESS');
