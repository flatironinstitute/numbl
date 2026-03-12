function result = compute_b(obj)
% compute_b: also calls a local helper named "localHelper" which returns value * 3
result = localHelper(obj.value);
end

function y = localHelper(x)
% Different localHelper — multiplies instead of adds
y = x * 3;
end
