####
#### benchmarks
####

@testset "printing seconds" begin
    @test repr(E._PrintSeconds(5e-9)) == "5.0ns"
    @test repr(E._PrintSeconds(4e-6)) == "4.0μs"
    @test repr(E._PrintSeconds(3e-3)) == "3.0ms"
    @test repr(E._PrintSeconds(2)) == "2.0s"
end

@testset "benchmarks" begin
    P = SometimesFails(LinearProblem(; n_x = 3, n_y = 4), 0.01)
    b = E.benchmark_and_stresstest(P; count = 1000)
    @test 0 < length(b.implicit_solve_errors) ≤ 20
    @test 0 < length(b.calculate_∂y∂x_errors) ≤ 20
end
