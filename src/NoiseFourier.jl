module NoiseFourier

include("FourierBasis.jl")
include("Fourier1DMatrix.jl")

using .Fourier1D
export assemble_matrix, noise_matrix, truncation_error, rigorous_norm 
export evaluate_trig_polynomial, abscissas

end
