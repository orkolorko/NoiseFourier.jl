
T(x) = 4*x*(1-x)
P = NoiseFourier.Fourier1D.assemble_matrix(T, 512; x_0 = 0, x_1 = 1)
D = NoiseFourier.Fourier1D.noise_matrix(0.1, 512)
M = D*P
v2, vinf = NoiseFourier.Fourier1D.rigorous_norm(M)
