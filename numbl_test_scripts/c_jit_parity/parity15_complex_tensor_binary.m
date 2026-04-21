% C-JIT parity gap #15: complex tensor element-wise binary ops.
%
% The C-JIT historically bailed on any complex tensor with
%   "complex tensor / unsupported ndim"
% in src/numbl-core/jit/c/cFeasibility.ts. Phase 2 wires
% numbl_complex_binary_elemwise / numbl_complex_scalar_binary_elemwise
% through for Add/Sub/Mul/ElemMul/Div/ElemDiv with mixed real/complex
% operands. NULL imag on the real side avoids a zero-fill.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> values below
%   numbl --opt 2 run <this>                         -> values below
%   numbl --opt 2 --check-c-jit-parity run <this>    -> values below
%   matlab -batch parity15_complex_tensor_binary     -> values below

function parity15_complex_tensor_binary()
    a = [1.0, 2.0, 3.0, 4.0];

    % 1) Real tensor + complex scalar = complex tensor, looped so the
    %    whole body goes through the C-JIT.
    r1 = 0.0; i1 = 0.0; r4 = 0.0;
    for k = 1:5
        b = a + (1 + 2i);          % complex tensor binary
        br = real(b);
        bi = imag(b);
        r1 = r1 + br(1);
        i1 = i1 + bi(1);
        r4 = r4 + br(4);
    end
    disp(r1 / 5)     % 2
    disp(i1 / 5)     % 2
    disp(r4 / 5)     % 5

    % 2) Complex tensor * complex tensor via conj.
    s_re = 0.0; s_im = 0.0;
    for k = 1:5
        z = a + a * 1i;            % complex tensor
        w = z .* conj(z);          % |z|^2, imag all zero
        wr = real(w);
        wi = imag(w);
        s_re = s_re + wr(3);       % 2*3^2 = 18
        s_im = s_im + wi(3);       % 0
    end
    disp(s_re / 5)   % 18
    disp(s_im / 5)   % 0

    % 3) Complex tensor / real scalar.
    d_re = 0.0; d_im = 0.0;
    for k = 1:5
        z = a + a * 1i;
        c = z / 2;
        cr = real(c);
        ci = imag(c);
        d_re = d_re + cr(3);       % 1.5
        d_im = d_im + ci(3);       % 1.5
    end
    disp(d_re / 5)   % 1.5
    disp(d_im / 5)   % 1.5

    disp('SUCCESS')
end
