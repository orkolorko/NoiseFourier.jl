using FFTW

module Fourier2D
export assemble_2D_matrix


"""
This is the function that maps frequencies i, j to a (2*Nx+1)*(2*Ny+1) vector
mirroring the behaviour of the reshape function, i.e., transforming a matrix 
(n, n) into a vector n*n by stacking the columns.
This means that, supposing Nx = 2, Ny =2 we have that
(0, 0) -> 1
(-1, 0) -> 5
(1, 0) -> 2
(0, 1) -> 6
(0, 2) -> 11
(0, -1) - > 21
(-1, -1) -> 25
"""
two_dimensional_index(i, j, Nx, Ny) = (unidimensional_index(j, Ny)-1)*(2*Nx+1)+unidimensional_index(i, Nx)

"""
This function takes a FFT computed on a grid of size Mx, My, and 
outputs the coefficients obtained by restricting the frequencies to 
[-Nx, Nx] × [-Ny, Ny] 
"""
function restrictfft!(new::Matrix, orig::Matrix, Nx, Ny)
    FFTNx , FFTNy = size(orig)
    new[1:Nx+1, 1:Ny+1] = @view orig[1:Nx+1, 1:Ny+1] # [0,...Nx] × [0,..., Ny]
    new[Nx+2:2*Nx+1, 1:Ny+1] = @view orig[FFTNx-Nx+1:FFTNx, 1:Ny+1] # [-Nx,...,-1] × [0,..., Ny]
    new[1:Nx+1, Ny+2:2*Ny+1] = @view orig[1:Nx+1, FFTNy-Ny+1:FFTNy] # [0,...,Nx] × [-Ny,..., -1]
    new[Nx+2:2*Nx+1, Ny+2:2*Ny+1] = @view orig[FFTNx-Nx+1:FFTNx, FFTNy-Ny+1:FFTNy] # [-Nx,...,-1] × [-Ny,..., -1]
    return new
end

function extendfft(v::Matrix{T}, FFTNx, FFTNy) where {T}
    Mx, My = size(v)
    Nx = (Mx-1)÷ 2
    Ny = (My-1)÷ 2

    w = zeros(T, (FFTNx, FFTNy))
    w[1:Nx+1, 1:Nx+1] = @view v[1:Nx+1, 1:Ny+1] 
    w[FFTNx-Nx+1:FFTNx, 1:Ny+1] = @view v[Nx+2:2*Nx+1, 1:Ny+1]
    w[1:Nx+1, FFTNy-Ny+1:FFTNy] = @view v[1:Nx+1, Ny+2:2*Ny+1]
    w[FFTNx-Nx+1:FFTNx,FFTNy-Ny+1:FFTNy] = @view v[Nx+2:2*Nx+1, Ny+2:2*Ny+1]
    return w
end

using LinearAlgebra, SparseArrays
function assemble_2D_matrix(F, Nx, Ny; FFTNx = 1024, FFTNy = 1024, x_0 = -1, x_1 = 1, y_0 = -1, y_1 =1, ϵ = 2^-30) 
    dx = [x_0+i*(x_1-x_0)/FFTNx for i in 0:FFTNx-1]; 
    dy = [y_0+i*(y_1-y_0)/FFTNy for i in 0:FFTNy-1]; 
    # probably better to implement this as an iterator
    
    Lx = abs(x_1-x_0)
    Ly = abs(y_1-y_0)

    one = [1 for (x, y) in Base.Iterators.product(dx, dy)]
    P = plan_fft(one)
    
    Fx = [F(x, y) for x in dx, y in dy]

    N = (2*Nx+1)*(2*Ny+1) # we are taking Nx positive and negative frequencies and the 0 frequency
    
    M = sparse(zeros(Complex{Float64},(N, N)))
    
    observablevalue = zeros(Complex{Float64}, (FFTNx, FFTNy)) 
    twodtransform = zeros(Complex{Float64}, (FFTNx, FFTNy))
    new = zeros(Complex{Float64}, (2*Nx+1, 2*Ny+1))

    for i in 1:2*Nx+1
        for j in 1:2*Ny+1
            
            l = inverse_unidimensional_index(i, Nx) # the index in the form [0, ..., Nx, -Nx, ..., -1]
            m = inverse_unidimensional_index(j, Ny) # the index in the form [0, ..., Nx, -Nx, ..., -1]
            for (ind, val) in pairs(Fx)
                observablevalue[ind[1], ind[2]] = ψ(l, m, val[1], val[2]; Lx= Lx, Ly =Ly) 
            end
            mul!(twodtransform, P, observablevalue)
            restrictfft!(new, twodtransform, Nx, Ny)
            
            for (ind, val) in pairs(new)
                val = purge(new[ind[1], ind[2]]/(FFTNx*FFTNy))
                
                μ, ν = inverse_unidimensional_index(ind[1], Nx), inverse_unidimensional_index(ind[2], Ny)

                if abs(val)!= 0
                    M[two_dimensional_index(l, m, Nx, Ny), two_dimensional_index(μ, ν, Nx, Ny)] = val
                end
            end
        end
    end
   # we take the adjoint since we are computing the adjoint operator
    return sparse(M)
end

end