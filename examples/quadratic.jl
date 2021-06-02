T(x) = 4*x*(1-x)
P = NoiseFourier.Fourier1D.assemble_matrix(T, 128; x_0 = 0, x_1 = 1)
D = noise_matrix(0.1, 128)
M = D*P
v2, vinf = rigorous_norm(M)
