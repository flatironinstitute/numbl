classdef NrClass
    % Methods that call the classdef file's local function nr_loc by name,
    % by handle and through cellfun/arrayfun.
    methods
        function r = viaFeval(obj)
            r = feval('nr_loc', 4);
        end
        function r = viaFevalString(obj)
            r = feval("nr_loc", 6);
        end
        function r = viaHandle(obj)
            h = @nr_loc;
            r = h(5);
        end
        function r = viaCellfunHandle(obj)
            r = cellfun(@nr_loc, {1, 2, 3});
        end
        function r = viaArrayfunHandle(obj)
            r = arrayfun(@nr_loc, [1 2]);
        end
        function r = viaCellfunLegacyName(obj)
            r = cellfun('isempty', {[], 1});
        end
        function r = existLocal(obj)
            r = exist('nr_loc');
        end
        function r = whichLocalFound(obj)
            r = ~isempty(which('nr_loc')) && ~isempty(which('nr_loc', '-all'));
        end
        function r = existMethods(obj)
            % Methods are not visible to exist by bare name.
            r = [exist('viaFeval'), exist('twice')];
        end
        function r = narginLocal(obj)
            r = nargin('nr_loc');
        end
    end
    methods (Static)
        function r = twice(x)
            r = 2 * x;
        end
    end
end

function r = nr_loc(x)
    r = 10 * x;
end
