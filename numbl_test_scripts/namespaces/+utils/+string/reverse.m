function result = reverse(s)
  result = '';
  for i = length(s):-1:1
    result = [result, s(i)];
  end
end
