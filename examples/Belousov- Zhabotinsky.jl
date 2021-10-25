using ArbNumerics, Plots
prec = setprecision(BigFloat, 256)

#a = BigFloat(0.506073569036822351319599371053047956980141736828203749380990114218225638827)
#b = BigFloat(0.02328852830307032054478158044023918735669943648088852646123182739831022528)
#c = BigFloat(0.121205692738975111744666848150620569782497212127938371936404761693002104361)

#function T(x)
#    if x <= (0.3)
#        return (a+cbrt((x-(1/8))))*exp(-x) + b
#    end
#    return c*(10*x*exp((-10*x)/3))^19 + b
#end
##
#epsilonc = BigFloat(0.7)
#tauc = BigFloat(0.8)

#function T(x)
#    t = x-(epsilonc /(2*pi))*sin(2*pi*x)+tauc
#    if t < 1
#        return t
#    else
#        return t-1
#    end
#end

a = BigFloat(3.93)
function T(x)
   t = a*x*(1-x)
    if t < 1
        return t
    else
        return t -1
    end
end

T(0.2)
#typeof(T(0.2))
##

Size = 256
noise_size = 0.05

P, gamma = NoiseFourier.Fourier1D.assemble_matrix(T, Size; x_0 = 0, x_1 = 1, T = BigFloat)
##
gamma

##
D = NoiseFourier.Fourier1D.noise_matrix(noise_size, Size)
M = D*P
typeof(M)
##
maxco = 0
for i in 1:Size
    if D[i,i]> maxco
        maxco = D[i,i]
    end
end

function Gamma(sigma,n)
    setprecision(256)
    return BigFloat((exp((-n^2*sigma^2)/2))/(sigma*n*sqrt(2*pi)))
end

#Approximation Error
gamma1 = gamma * maxco + Size*Gamma(noise_size, 2048)
typeof(gamma1)
##

#Discretization Error
prec = setprecision(BigFloat, 64)
P1, asd = NoiseFourier.Fourier1D.assemble_matrix(T, Size; x_0 = 0, x_1 = 1, T = Float64)
##
D1 = NoiseFourier.Fourier1D.noise_matrix(noise_size, Size)
M1 = D1*P1
v2, vinf = NoiseFourier.Fourier1D.rigorous_norm(M1, k=500)
##
#print(asd)
print(v2)
##

function discr_error(v)
    setprecision(256)
    Ci = BigFloat(0.0)
    t = size(v)[1]
    for i in 1:t-1
        Ci += v[i]
    end
    return (1/(1-v[t])) * Ci * (1 + 1/(noise_size*sqrt(2*pi)) + Gamma(noise_size,Size)) * Gamma(noise_size, Size)
end

##
discr_error(v2)
##

#Power method
prec = setprecision(BigFloat, 256)
v = zeros(2*Size+1)
v[1] = BigFloat(1.0)
for i in 1:1000
    @info(i)
    v = M * v
    v /= norm(v,2)
end
lambda = abs(adjoint(v)*M*v)
##
norm(M*v -v,2)
##
#Eigen error
epsilon = BigFloat(norm(M*v -v,2))
##
eigenerr = epsilon + gamma1/2

##
#Plot
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

p = 50000
x = range(0,1, length = p)
y = zeros(p)
for i in 1:p
    y[i] = measure(x[i],v)
end
plot(x,y,ylims=(0,5),fmt = :png)#,xticks=25:5:75)

##
#Total error
#err = discr_error(v2) + gamma1 + eigenerr
##
print(gamma1)

##
print(discr_error(v2))
##
print(eigenerr)