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

function norm_2_estimator_nonrig(M; n_it= 10)
    n, m = size(M)
    @assert n==m
    v = ones(n-1)
    N = M[2:end, 2:end]

    start_norm =  norm(v, 2)
    
    for i in 1:n_it
        v = N'*(N*v)
        v = v/norm(v,2)
    end
    w = v

    v = N*v
    
    return abs(v'*v), w/norm(w,2)
end    

using IntervalArithmetic, IntervalRootFinding

widen(x::T, fatten) where {T} = x+2.0^(-fatten)*Interval{T}(-1, 1)
widen(x::Complex{T}, fatten) where {T} = x+2.0^(-fatten)*(Interval{T}(-1, 1)+im*Interval{T}(-1, 1))

function Yamamoto_certify(M, λ, v; fatten = 10)
    N = M[2:end, 2:end]
    F(λ, v) = [N'*N*v-λ*v; v'*v-1]
    v_fat = widen.(v, fatten)
    lam = widen(λ, fatten)
    
    DF(λ, v) = [N'*N-λ*I -v;
                v'          0]

    w = [λ; v] - DF(lam, v_fat)\F(λ, v)
    @info w[1], lam
    return (abs(w[1])).hi
end

function RigorousNorm(M; k = 10)
   A = M
   n, m = size(M)
   norms = zeros(Float64, k)
   for i in 1:k
       λ, v = norm_2_estimator_nonrig(A)
       @info λ
       norms[i] = Yamamoto_certify(A, λ, v; fatten = 100)
       A = M*A
   end     
   return norms, sqrt(n-1)*norms #check to make it rigorous
end

end