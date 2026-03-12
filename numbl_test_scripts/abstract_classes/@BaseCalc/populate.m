function [obj, r] = populate(obj, x, y)
    % This method is defined in the parent class (BaseCalc).
    % It calls obj.compute(x, y) where compute is an abstract static method.
    % The actual dispatch should go to the concrete subclass's implementation.
    r = obj.compute(x, y);
end
