@testset "∂Y∂X pretty printing" begin
    repr(E.calculate_∂y∂x(LinearProblem(; n_x = 3, n_y = 4), randn(3), randn(4))) == "∂Y∂X(« 4 × 4 »)"
end

@testset "sanity checks failure" begin
    checks = E.API_sanity_checks(SometimesFails(LinearProblem(; n_x = 3, n_y = 4), 1.1))
    @test repr(checks) isa AbstractString
    @test !checks.all_ok
end
