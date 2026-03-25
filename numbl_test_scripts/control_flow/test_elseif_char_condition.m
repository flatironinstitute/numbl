% Test that elseif with char conditions evaluates correctly.
% Empty char '' is falsy, non-empty char 'a' (=97) is truthy.

function y = elseif_empty_char()
  x = 0;
  if false
    x = 1;
  elseif ''
    x = 2;
  end
  y = x;
end

function y = elseif_nonempty_char()
  x = 0;
  if false
    x = 1;
  elseif 'a'
    x = 2;
  end
  y = x;
end

r1 = elseif_empty_char();
assert(r1 == 0, 'elseif with empty char should be falsy');

r2 = elseif_nonempty_char();
assert(r2 == 2, 'elseif with nonempty char should be truthy');

disp('SUCCESS');
