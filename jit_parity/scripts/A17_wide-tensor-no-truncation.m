% DIAGNOSIS: opt0 interpreter truncates matrix display at >20 cols
%   ("Columns 1 through N" header + "..." ellipsis, via format2DSlice);
%   opt1 JS-JIT's mtoc2_disp_tensor has NO truncation -> prints full row.
% --opt 0:
%     Columns 1 through 25
%
%        1   2   3 ... 10   ...   16 ... 25
% --opt 1:
%        1   2   3 ... 25   (full, no header/ellipsis)
v = zeros(1,25);
for k=1:25
  v(k) = k;
end
disp(v)
