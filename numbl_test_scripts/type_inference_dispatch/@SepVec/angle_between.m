function r = angle_between(obj, other)
    % Separate-file method calling other separate-file methods
    d = obj.dot_product(other);
    m1 = obj.magnitude();
    m2 = other.magnitude();
    r = acos(d / (m1 * m2));
end
