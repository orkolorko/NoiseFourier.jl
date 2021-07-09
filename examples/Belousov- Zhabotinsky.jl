using ArbNumerics
setextrabits(0)
prec  = 300
setprecision(ArbComplex, bits = prec)
a = ArbComplex(0.506073569036822351319599371053047956980141736828203749380990114218225638827)
b = ArbComplex(0.02328852830307032054478158044023918735669943648088852646123182739831022528)
c = ArbComplex(0.121205692738975111744666848150620569782497212127938371936404761693002104361)

function T(x)
    if x <= (0.3)
        return (a+cbrt((x-(1/8))))*exp(-x) + b
    end
    return c*(10*x*exp((-10*x)/3))^19 + b
end

T(0.2)
typeof(T(0.2))
##

Size = 128
noise_size = 0.05

P = NoiseFourier.Fourier1D.assemble_matrix_high_precision(T, Size; x_0 = 0, x_1 = 1)
##
D = NoiseFourier.Fourier1D.noise_matrix(noise_size, Size)
M = D*P

v2, vinf = NoiseFourier.Fourier1D.rigorous_norm(M, k=500)

print(vinf)


##
function convert_to_bigfloat(M, n, m)
    setprecision(1000)
    Q = zeros(Complex{Float64},n,m)
    ze = 0
    zer = []
    max_nnz_rows = 0
    a = 0
    b = 0
    for i in 1:n
        for j in 1:m
            a = BigFloat(real((M[i,j])))
            b = BigFloat(imag((M[i,j])))
            Q[i,j] = a + im*b
            if Q[i,j] == 0
                ze = ze +1
            end
        end
        push!(zer, ze)
        ze = 0
    end
    max_nnz_rows = maximum(zer)
    return Q#, max_nnz_rows
end

function idx_to_freq(j,k)
    # k is the size of the matrix
    if j>=0 && j <=k ÷ 2+1
        return j-1
    end
    if j >k ÷ 2+1
        return j-k-1
    end
end 

function measure(x,v)
    t = real(v[1])
    for i in 2:Size
        a = 2*real(v[i])   
        b = 2*imag(v[i]) 
        k = idx_to_freq(i,Size)
        t += a*cos(2*pi*x*k) - b*sin(2*pi*x*k)
    end
    return t
end

function Gamma(sigma,n)
    setprecision(1000)
    return BigFloat((exp((-n^2*sigma^2)/2))/(sigma*n*sqrt(2*pi)))
end


##

setprecision(1000)
Z = convert_to_bigfloat(M, Size, Size)
v = zeros(Size)
v = convert_to_bigfloat(v, Size, 1)
v[2]=BigFloat(1.0)
for i in 1:4000
    v = Z * v
end
epsilo = BigFloat(norm(Z*v - v, 2))
##

function appr_error_new(v)
    Ci = 0
    t = size(v)[1]
    for i in 1:t-1
        Ci += v[i]
    end
    return (1/(1-v[t])) * Ci * (1 + epsilo + Gamma(noise_size,Size) + 1/(noise_size*sqrt(2*pi))) * Gamma(noise_size, Size)
end

function dft_error(Size, sigma, N)
    return Size * Gamma(sigma,N)
end

p = 50000
x = range(0,1, length = p)
y = zeros(p)
for i in 1:p
    y[i] = measure(x[i],v)
end
plot(x,y,ylims=(0,5),fmt = :png)#,xticks=25:5:75)


@info dft_error(Size, noise_size, 2048)
@info appr_error_new(vinf)

##

