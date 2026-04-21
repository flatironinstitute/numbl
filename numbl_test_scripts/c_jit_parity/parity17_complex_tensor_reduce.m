% C-JIT parity gap #17: complex tensor reductions + abs(complex_tensor).
%
% Phase 3 wires:
%   sum(z), prod(z)     → complex scalar via numbl_complex_flat_reduce
%   any(z), all(z)      → real scalar, same kernel (flag in out_re)
%   abs(z)              → real tensor via numbl_complex_abs
%
% max / min / mean on complex still bail (the kernel returns
% NUMBL_ERR_BAD_OP for those — MATLAB itself restricts them).
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> values below
%   numbl --opt 2 run <this>                         -> values below
%   numbl --opt 2 --check-c-jit-parity run <this>    -> values below
%   matlab -batch parity17_complex_tensor_reduce     -> values below

function parity17_complex_tensor_reduce()
    a = [1.0, 2.0, 3.0, 4.0];

    % 1) sum on a complex tensor (returns complex scalar).
    s_re = 0.0; s_im = 0.0;
    for k = 1:5
        z = a + a * 1i;          % 1+i, 2+2i, 3+3i, 4+4i
        s = sum(z);              % 10 + 10i
        s_re = s_re + real(s);
        s_im = s_im + imag(s);
    end
    disp(s_re / 5)   % 10
    disp(s_im / 5)   % 10

    % 2) prod on a small complex tensor.
    b = [1.0, 2.0];
    p_re = 0.0; p_im = 0.0;
    for k = 1:5
        z = b + b * 1i;          % (1+i)(2+2i) = 0 + 4i
        p = prod(z);
        p_re = p_re + real(p);
        p_im = p_im + imag(p);
    end
    disp(p_re / 5)   % 0
    disp(p_im / 5)   % 4

    % 3) abs on complex tensor → real tensor.
    ab = 0.0;
    for k = 1:5
        z = a + a * 1i;
        mag = abs(z);            % sqrt(2) * a = [~1.41, 2.83, 4.24, 5.66]
        ab = ab + mag(3);        % 3*sqrt(2) ≈ 4.2426
    end
    disp(ab / 5)     % 3*sqrt(2) ≈ 4.2426

    disp('SUCCESS')
end
