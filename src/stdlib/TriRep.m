classdef TriRep < triangulation
    % (Not recommended) Triangulation representation. Provided for legacy
    % code; use triangulation instead. TriRep inherits freeBoundary,
    % vertexAttachments, etc. from triangulation and exposes the legacy
    % property names X (vertex coordinates) and Triangulation (connectivity).
    %
    %   TR = TriRep(tri, x, y)
    %   TR = TriRep(tri, x, y, z)
    %   TR = TriRep(tri, P)
    properties
        X
        Triangulation
    end
    methods
        function obj = TriRep(tri, varargin)
            obj = obj@triangulation(tri, varargin{:});
            obj.X = obj.Points;
            obj.Triangulation = obj.ConnectivityList;
        end
    end
end
