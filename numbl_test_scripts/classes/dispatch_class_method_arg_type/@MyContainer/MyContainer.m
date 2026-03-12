classdef MyContainer
    properties
        items
    end
    methods
        function obj = MyContainer(items)
            if nargin > 0
                obj.items = items;
            else
                obj.items = {};
            end
        end
        function out = isempty(obj)
            % Custom isempty: container is empty if items is empty
            out = isempty(obj.items);
        end
        function out = doCheck(obj, varargin)
            % Call isempty on varargin (a cell array) inside a class method.
            % This should use the builtin isempty, not @MyContainer/isempty.
            out = isempty(varargin);
        end
    end
end
