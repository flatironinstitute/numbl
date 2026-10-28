classdef (Abstract) AttrAbstract
    % Fixture for class_access.m: abstract members and a sealed method.
    properties (Abstract)
        ap
    end
    methods (Abstract)
        r = doIt(obj)
    end
    methods (Sealed)
        function v = sealedM(obj)
            v = 1;
        end
    end
end
