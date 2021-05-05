using LinearAlgebra, SparseArrays

module Fourier1D
using ..FourierBasis: inverse_unidimensional_index, ϕ, purge
export assemble_1D_matrix, noise_matrix, truncation_error

function restrictfft!(new::Vector, orig::Vector, Nx)
    FFTNx = length(orig)
    new[1:Nx+1] = @view orig[1:Nx+1]
    new[Nx+1:2*Nx+1] = @view orig[FFTNx-Nx:FFTNx]
    return new
end 

function extendfft(orig::Vector, FFTNx) 
    N = length(orig)
    Nx = (N-1)÷ 2 
    return [orig[1:Nx]; zeros(Complex{Float64}, FFTNx-N); orig[Nx+1:end]]
end

using FFTW, SparseArrays, LinearAlgebra

function assemble_1D_matrix(T, Nx; FFTNx = 1024, x_0 = 0, x_1 = 1, ϵ = 2^-30) 
    
    dx = [x_0+i*(x_1-x_0)/FFTNx for i in 0:FFTNx-1]; 
    
    Lx = abs(x_1-x_0)
    
    one = [1 for x in dx]
    P = plan_fft(one)
     # we take the adjoint since we are computing the adjoint operator
  
    Tx = [T(x) for x in dx]

    N = (2*Nx+1) # we are taking Nx positive and negative frequencies and the 0 frequency
    
    M = zeros(Complex{Float64},(2*Nx+1, 2*Nx+1))
    
    observablevalue = zeros(Complex{Float64}, FFTNx) 
    onedtransform = zeros(Complex{Float64}, FFTNx)
    new = zeros(Complex{Float64}, 2*Nx+1)

    for i in 1:2*Nx+1    
        l = inverse_unidimensional_index(i, Nx) # the index in the form [0, ..., Nx, -Nx, ..., -1]
        
        for (ind, val) in pairs(Tx)
            observablevalue[ind] = ϕ(l, val; L=Lx) 
        end
        mul!(onedtransform, P, observablevalue)
        restrictfft!(new, onedtransform, Nx)
            
        for (ind, val) in pairs(new)
                if abs(val)!= 0
                    # this is the adjoint matrix of the Koopman operator
                    M[i, ind] = purge(conj(val)/FFTNx, ϵ)
                end
        end
    end
    return M
end

noise_matrix(Nx, σ) = Diagonal([[exp(-(k*σ)^2/2) for k in 0:Nx]; [exp(-(k*σ)^2/2) for k in -Nx:-1]])

truncation_error(Nx, σ) = exp(-(Nx*σ)^2/2)/(Nx*σ*sqrt(2*pi))

#function norm_bound(Q)
#    T = Hermitian(Q'*Q)    
#end
    


end