function h = make_dog_helper_()
% Helper: output variable h is uninitialized, then assigned via h(1) = obj.
% In MATLAB this works because h is truly undefined. In numbl the codegen
% initializes h = 0 which must not prevent the class_instance assignment.
    d = Dog_('Rex');
    h(1) = d;
end
