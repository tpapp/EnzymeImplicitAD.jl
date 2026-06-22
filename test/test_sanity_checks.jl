@testset "sanity checks failure" begin
    checks = E.API_sanity_checks(SometimesFails(LinearProblem(; n_x = 3, n_y = 4), 1.1))
    @test repr(checks) isa AbstractString
    @test !checks.all_ok
    @test checks.check_eltype ≡ nothing
end
