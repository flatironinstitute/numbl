classdef SimpleContainer
    properties
        value
        data
        items
    end
    methods
        function obj = SimpleContainer()
            obj.value = 0;
            obj.data = struct();
            obj.items = {};
        end
        function obj = negateItems(obj)
            % Mirrors chebfun uminus pattern:
            % F(j).prop = ...; F(j).items{k} = ...
            for j = 1:numel(obj)
                obj(j).value = -obj(j).value;
                for k = 1:numel(obj(j).items)
                    obj(j).items{k} = -obj(j).items{k};
                end
            end
        end
    end
end
