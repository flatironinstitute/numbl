classdef (Sealed) PropertyEvent < event.EventData
    % Data passed to PreSet/PostSet/PreGet/PostGet listener callbacks:
    % Source is the meta.property of the observed property and
    % AffectedObject the object whose property is accessed.
    properties (SetAccess = private)
        AffectedObject = []
    end
end
