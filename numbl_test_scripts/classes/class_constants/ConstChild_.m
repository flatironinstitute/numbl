classdef ConstChild_ < ConstHolder
    % Fixture for class_constants.m: inherits Constant properties.
    methods
        function obj = ConstChild_()
            obj@ConstHolder(0);
        end
    end
end
