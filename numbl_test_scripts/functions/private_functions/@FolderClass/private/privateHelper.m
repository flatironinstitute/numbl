function y = privateHelper(obj)
    % obj is a class instance, so obj.val has unknown type
    % This forces localDouble(obj.val) to have unknown arg types
    y = localDouble(obj.val) + 1;
end

function y = localDouble(x)
    y = x * 2;
end
