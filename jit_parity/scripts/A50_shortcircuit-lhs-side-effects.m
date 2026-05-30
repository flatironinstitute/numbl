% DIAGNOSIS: lowering `A && B` / `A || B` (lower.ts lowerBinary) pulls `exact`
% from the lowered LHS and, when it short-circuits the result, returns a bare
% NumLit -- DISCARDING the already-lowered LHS. The LHS of a short-circuit is
% ALWAYS evaluated (only the RHS may be skipped), so dropping a side-effecting
% LHS is wrong. A user-function call like lhs(-1) gets constant-folded (its
% return type carries `exact`) while still containing the fprintf, so opt1/opt2
% silently drop the call.
%
% --opt 0 output (correct):  "lhs(-1) lhs(3) b1=0 b2=1"
% --opt 1/2 output (buggy):  "b1=0 b2=1"   (both lhs() calls vanished)
b1 = lhs(-1) && lhs(2); % LHS false -> RHS skipped, but lhs(-1) MUST run
b2 = lhs(3) || lhs(4); % LHS true  -> RHS skipped, but lhs(3) MUST run
fprintf('b1=%d b2=%d\n', b1, b2);

function r = lhs(x)
    fprintf('lhs(%d) ', x);
    r = x > 0;
end
