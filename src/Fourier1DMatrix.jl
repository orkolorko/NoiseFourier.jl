using LinearAlgebra, SparseArrays, IntervalArithmetic, IntervalRootFinding, Plots

module Fourier1D
using ..FourierBasis: inverse_unidimensional_index, ϕ, purge
export assemble_matrix, noise_matrix, truncation_error, rigorous_norm 
export evaluate_trig_polynomial, abscissas

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
"""
assemble_matrix(T, Nx; FFTNx, x_0, x_1, ϵ)

This function assembles the matrix of the transfer operator of T using the FFT

Arguments:
- T : the dynamic
- Nx : the truncation frequence
- FFTNx : the number of points used to compute the FFT
- x_0 : left endpoint of the domain of T, defaults to 0
- x_1 : right endpoint of the domain of T, defaults to 1
- ϵ : the purge threshold, under this threshold the entries are set to 0
"""
function assemble_matrix(T, Nx; FFTNx = 2048, x_0 = 0, x_1 = 1, ϵ = 2^-30) 
    
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

"""
noise_matrix(σ, Nx)

returns the matrix of the (periodic) convolution operator with a Gaussian of average 0
and variance σ in the Fourier basis, truncated at frequence Nx 
(Check what happens when we change x_0 and x_1, probably rescaling σ)

"""
noise_matrix(σ, Nx) = Diagonal([[exp(-(k*σ)^2/2) for k in 0:Nx]; [exp(-(k*σ)^2/2) for k in -Nx:-1]])

"""
truncation_error(σ, Nx)

This function returns the truncation error when truncating frequencies at Nx
when the Gaussian has variance σ
(check the rescaling)
"""
truncation_error(σ, Nx) = exp(-(Nx*σ)^2/2)/(Nx*σ*sqrt(2*pi))


"""
This function computes a candidate for the top singular value
of M|_{V_0} square, i.e., it returns λ^2 and the associated singular vector 
"""
function norm_2_estimator_square_nonrig(M; n_it= 10)
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

    w = [lam; v_fat]
    w0 = [λ; v] - DF(lam, v_fat)\F(λ, v)
    
    
    w = [λ; v] - DF(w0[1], w0[2:end])\F(λ, v)
    
    #t = intersect( abs(w0[1]), abs(w[1]))
    
    return (abs(w[1])).hi
end

"""
This function uses the classical bound, cited in

RESIDUAL BOUNDS ON APPROXIMATE EIGENSYSTEMS OF NONNORMAL MATRICES*
W. KAHAN, B. N. PARLETT AND E. JIANGt

Theorem 1

"""
function faster_certify(A, λ, v)
    N = A[2:end, 2:end]
    w = Interval.(v)
    ν = Interval(λ)
    #@info λ
    ρ = norm(N'*N*w-ν*w, 2)/norm(w, 2)
    ν = hull(-ρ, ρ) + λ
    return ν
end

using IntervalArithmetic

upper(x::Interval) = x.hi

"""
rigorous_norm(M; k = 10)

Computes the L² and L∞ norm of the powers of M|_V₀ up to the power k.
The algorithm first computes a numeric estimate of the top Singular Value
and then certifies it by an Interval Newton step

Outputs: (v2, v∞)
- v₂ : it is the vector with the rigorous bounds on the L² norm
- v∞ : it is the vector with the rigorous bounds on the L∞ norm;
       these are obtained by observing that
       ||f||₂ = ||w||₂, 
       where w is the vector of the Fourier coefficients of the trigonometric polynomial f
       ||f||₂ <= ||f||∞ <= ∑ |w_i| <= √n*||f||₂  
       so
       ||M||∞ <= √n *||M||₂
"""
function rigorous_norm(M; k = 1000)
   A = M
   n, m = size(M)
   norms = zeros(Interval{Float64}, k)
   for i in 1:k
       λ, v = norm_2_estimator_square_nonrig(A)
       #@info λ
       #norms[i] = Yamamoto_certify(A, λ, v; fatten = 100)
       norms[i] = sqrt(faster_certify(A, λ, v))
       A = M*A
   end     
   rescale = sqrt(Interval(n-1))
   @info "rescale", rescale
   return norms, rescale*norms #check to make it rigorous
end

"""
evaluate_trig_polynomial(v; FFTNx = 16384)

Extends a vector v to size FFTNx (padding with zeros), and then takes the real part of the
inverse fourier transform, this gives us the value of IFFT(v) at FFTNx points.
Useful for plots... beware the fact that when plotting you need to give abscissas
"""
evaluate_trig_polynomial(v; FFTNx = 16384) = real.(ifft(extendfft(v, FFTNx)))

"""
Returns the abscissas for plotting, i.e., FFTNx equispaced points between x_0 and x_1
"""
abscissas(x_0, x_1, FFTNx) = [x_0+(i-1)*(x_1-x_0)/FFTNx for i in 1:FFTNx]



end