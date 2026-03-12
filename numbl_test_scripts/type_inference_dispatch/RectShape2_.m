classdef RectShape2_ < BaseShape2_
    properties
        Width
        Height
    end
    methods
        function obj = RectShape2_(w, h, color)
            if nargin < 3
                color = 'black';
            end
            obj = obj@BaseShape2_(color);
            obj.Width = w;
            obj.Height = h;
        end
        function r = area(obj)
            r = obj.Width * obj.Height;
        end
        function r = describe(obj)
            % Override base describe
            r = 1;
        end
    end
end
