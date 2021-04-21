using NoiseFourier
using Test

@testset "NoiseFourier.jl" begin
    N = 1024
    @test [NoiseFourier.unidimensional_index(i, N) for i in 0:N] == [j for j in 1:N+1]
    @test [NoiseFourier.unidimensional_index(i, N) for i in -N:-1] == [j for j in N+2:2*N+1]

    for i in 1:2049
        @test NoiseFourier.unidimensional_index(NoiseFourier.inverse_unidim_index(i, N), N) == i
    end
end
