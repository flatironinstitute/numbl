classdef PulseqPatterns_ < handle
    % Helper for tests of patterns used by Pulseq-style code:
    % multi-output methods called without parens, and chained
    % indexing into cell properties inside methods.
    properties
        c = {};
    end
    methods
        function [a, b] = two(obj)
            a = 1;
            b = 2;
        end

        function v = lastPositive(obj, d, j)
            % find-derived (1x1 tensor) index into a cell property,
            % then paren-index the extracted element
            [~, k] = find(d > 0, 1, 'last');
            v = obj.c{k}(j);
        end
    end
end
