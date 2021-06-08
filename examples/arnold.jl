epsilon = 0.7
tau = 1.4
T(x) = x-(epsilon/(2*pi))*sin(2*pi*x)+tau
P = NoiseFourier.Fourier1D.assemble_matrix(T, 512; x_0 = 0, x_1 = 1)
D = NoiseFourier.Fourier1D.noise_matrix(0.1, 512)
M = D*P
v2, vinf = NoiseFourier.Fourier1D.rigorous_norm(M)
