function result = compute(obj, x)
% Primary method: calls a local helper function defined in this same file
result = applyOffset(obj.value, x);
end

function y = applyOffset(base, offset)
% Local helper — only visible within this file
y = base + offset;
end
