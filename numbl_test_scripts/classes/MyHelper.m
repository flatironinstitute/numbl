classdef MyHelper
    methods (Static)
        function [x, y] = grab_first_two(varargin)
            x = varargin{1};
            if nargin >= 2
                y = varargin{2};
            else
                y = [];
            end
        end
    end
end
