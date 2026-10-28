classdef AttrPrivCtor
    % Fixture for class_access.m: a private constructor reached through a
    % static factory.
    properties
        v = 0
    end
    methods (Access = private)
        function obj = AttrPrivCtor(v)
            obj.v = v;
        end
    end
    methods (Static)
        function obj = make(v)
            obj = AttrPrivCtor(v);
        end
    end
end
