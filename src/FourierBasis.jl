module FourierBasis


@inline ϕ(k, x; L=1)  = exp(im*2*pi*x*k/L) # L is the length of the interval
@inline ψ(k, h, x, y; Lx=1, Ly=1) = (-1)^(k+h)*ϕ(k, x; L = Lx)*ϕ(h, y; L= Ly)

purge(x, ϵ=2^-10) = abs(x)<ϵ ? 0.0 :  x
purge(x::Complex)= purge(real(x))+im*purge(imag(x))

make_dx(N) = [i/2^N for i in 0:2^N-1]

# converts [0,1,..., Nx,-Nx,...,-1] to [1,...,2*Nx+1]
unidimensional_index(i, Nx) = i>=0 ? i+1 : 2*Nx+2+i 
inverse_unidimensional_index(i, Nx) = 1<=i<=Nx+1 ? i-1 : i-2*Nx-2 

end