function r = cross_product(obj, other)
    cx = obj.Y * other.Z - obj.Z * other.Y;
    cy = obj.Z * other.X - obj.X * other.Z;
    cz = obj.X * other.Y - obj.Y * other.X;
    r = SepVec(cx, cy, cz);
end
