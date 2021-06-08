C = 0.1
A = 0.07
T(x) = x+C+A*sin(4*pi*x)
P = NoiseFourier.Fourier1D.assemble_matrix(T, 512; x_0 = 0, x_1 = 1)
D = NoiseFourier.Fourier1D.noise_matrix(0.1, 512)
M = D*P
v2, vinf = NoiseFourier.Fourier1D.rigorous_norm(M)
