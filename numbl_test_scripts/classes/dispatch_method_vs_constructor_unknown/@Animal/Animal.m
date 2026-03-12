classdef Animal
    properties
        name
    end
    methods
        function obj = Animal(n)
            obj.name = n;
        end
        % Method with the same name as the Info class constructor.
        % When calling Info(animalObj), MATLAB should dispatch here,
        % not to the @Info constructor.
        function out = Info(obj)
            out = ['Animal: ' obj.name];
        end
    end
end
