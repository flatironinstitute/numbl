classdef SquareShape2_ < RectShape2_
    methods
        function obj = SquareShape2_(side, color)
            if nargin < 2
                color = 'black';
            end
            obj = obj@RectShape2_(side, side, color);
        end
        function r = describe(obj)
            % Override rect describe
            r = 2;
        end
        function r = diagonal(obj)
            r = obj.Width * sqrt(2);
        end
    end
end
