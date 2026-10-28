classdef AttrOverrideSealed < AttrConcrete
    % Fixture for class_access.m: overrides a Sealed method (invalid).
    methods
        function v = sealedM(obj)
            v = 2;
        end
    end
end
