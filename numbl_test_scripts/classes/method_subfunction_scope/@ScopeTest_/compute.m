function result = compute(obj)
% Primary method: sets n=5, calls subfunction helper, then uses n
% The subfunction also uses n as a local variable.
% In MATLAB, subfunctions have their own scope, so helper's n should
% NOT affect the primary method's n.
n = 5;
tmp = helper(obj.val);
% n should still be 5 here (not modified by helper)
result = tmp + n;
end

function y = helper(x)
% Local subfunction: uses n as a local variable
% This n should be completely independent of compute's n
n = 2;
y = x + n;
end
