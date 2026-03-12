classdef ChainedCellAsgnObj_
    properties
        items
    end
    methods
        function obj = ChainedCellAsgnObj_(n)
            obj.items = cell(1, n);
            for i = 1:n
                obj.items{i} = i * 10;
            end
        end
        function F = negateItems(F)
            % Pattern: F(j).items{k} = val  (compound V(i).field{k} = rhs)
            for j = 1:numel(F)
                for k = 1:numel(F(j).items)
                    F(j).items{k} = -F(j).items{k};
                end
            end
        end
        function obj = subsasgn(obj, S, val)
            % This should NOT be called for F(j).items{k} = val inside methods
            error('subsasgn should not be called');
        end
    end
end
