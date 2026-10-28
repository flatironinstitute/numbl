classdef EvLog < handle
    % Fixture for the events tests: records what listener callbacks saw.
    properties
        items = {}
    end
    methods
        function add(obj, s)
            obj.items{end+1} = s;
        end
        function clear(obj)
            obj.items = {};
        end
        function s = str(obj)
            s = strjoin(obj.items, ',');
        end
    end
end
