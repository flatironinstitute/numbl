% Test that one private function can call another private function
result = priv_a(3);
assert(result == 7);  % priv_a(3) = priv_b(3) + 1 = 6 + 1 = 7
disp('SUCCESS')
