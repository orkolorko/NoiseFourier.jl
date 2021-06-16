
##
a = (0.506073569036822351319599371053047956980141736828203749380990114218225638827)
b = (0.02328852830307032054478158044023918735669943648088852646123182739831022528)
c = (0.121205692738975111744666848150620569782497212127938371936404761693002104361)

function T(x)
    if x <= (0.3)
        return (a+cbrt((x-(1/8))))*exp(-x) + b
    end
    return c*(10*x*exp((-10*x)/3))^19 + b
end

T(0.2)
##
P = NoiseFourier.Fourier1D.assemble_matrix(T, 128; x_0 = 0, x_1 = 1)
D = NoiseFourier.Fourier1D.noise_matrix(0.005, 128)
M = D*P
v2, vinf = NoiseFourier.Fourier1D.rigorous_norm(M)

print(v2)
print(vinf)

function convert_to_float(M, n)
    Q = zeros(Complex{Float64},n,n)
    ze = 0
    zer = []
    max_nnz_rows = 0
    a = 0
    b = 0
    for i in 1:n
        for j in 1:n
            a = Float64(real((M[i,j])))
            b = Float64(imag((M[i,j])))
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
    for i in 2:128
        a = 2*real(v[i])   
        b = 2*imag(v[i]) 
        k = idx_to_freq(i,512)
        t += a*cos(2*pi*x*k) - b*sin(2*pi*x*k)
    end
    return t
end

Z = convert_to_float(M, 128)
v = eigvecs(Z)[:,128]
p = 50000
x = range(0,1, length = p)
y = zeros(p)
for i in 1:p
    y[i] = measure(x[i],v)
end
plot(x,y,ylims=(0,25),fmt = :png)#,xticks=25:5:75)