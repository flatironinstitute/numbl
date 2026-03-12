classdef BaseShape2_
    properties
        Color
    end
    methods
        function obj = BaseShape2_(color)
            if nargin >= 1
                obj.Color = color;
            else
                obj.Color = 'black';
            end
        end
        function r = describe(obj)
            r = 0;
        end
        function r = color_code(obj)
            if strcmp(obj.Color, 'red')
                r = 1;
            elseif strcmp(obj.Color, 'blue')
                r = 2;
            else
                r = 0;
            end
        end
    end
end
