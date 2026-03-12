function d = distance(obj, other)
  dx = obj.x - other.x;
  dy = obj.y - other.y;
  d = sqrt(dx * dx + dy * dy);
end
