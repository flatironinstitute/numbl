classdef triangulation
    % Triangulation representation providing topological queries over a
    % triangle (or tetrahedron) mesh. Mirrors a subset of MATLAB's
    % triangulation class.
    %
    %   TR = triangulation(tri, P)        % P is an n-by-d coordinate matrix
    %   TR = triangulation(tri, x, y)     % 2-D vertex coordinate columns
    %   TR = triangulation(tri, x, y, z)  % 3-D vertex coordinate columns
    properties
        Points
        ConnectivityList
    end
    methods
        function obj = triangulation(tri, varargin)
            if nargin == 0
                return;
            end
            obj.ConnectivityList = tri;
            if numel(varargin) == 1
                obj.Points = varargin{1};
            elseif numel(varargin) >= 2
                obj.Points = [varargin{:}];
            else
                error('triangulation: vertex coordinates are required');
            end
        end

        function [F, P] = freeBoundary(obj)
            % Free boundary facets: the edges referenced by exactly one
            % triangle, ordered into connected, consistently oriented loops.
            tri = obj.ConnectivityList;
            E = [tri(:, [1 2]); tri(:, [2 3]); tri(:, [3 1])];
            Es = sort(E, 2);
            [~, ~, ic] = unique(Es, 'rows');
            counts = accumarray(ic, 1);
            isB = counts(ic) == 1;
            bedges = E(isB, :);

            n = size(bedges, 1);
            F = zeros(n, 2);
            used = false(n, 1);
            pos = 1;
            while pos <= n
                startRow = find(~used, 1);
                if isempty(startRow)
                    break;
                end
                cur = startRow;
                loopStart = bedges(cur, 1);
                while true
                    F(pos, :) = bedges(cur, :);
                    used(cur) = true;
                    pos = pos + 1;
                    nextv = bedges(cur, 2);
                    if nextv == loopStart
                        break;
                    end
                    cand = find(~used & bedges(:, 1) == nextv, 1);
                    if isempty(cand)
                        break;
                    end
                    cur = cand;
                end
            end

            if nargout > 1
                vid = unique(F(:));
                P = obj.Points(vid, :);
                remap = zeros(max(vid), 1);
                remap(vid) = 1:numel(vid);
                F = remap(F);
            end
        end

        function V = vertexAttachments(obj, id)
            % IDs of the triangles attached to each vertex, returned as a
            % cell array with one row vector of triangle IDs per vertex.
            tri = obj.ConnectivityList;
            nv = size(obj.Points, 1);
            if nargin < 2
                id = (1:nv)';
            end
            id = id(:);
            V = cell(numel(id), 1);
            for k = 1:numel(id)
                V{k} = find(any(tri == id(k), 2))';
            end
        end
    end
end
