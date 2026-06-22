@testset "implicit solver" begin
    n_x, n_y = 4, 5
    P0 = LinearProblem(; n_y, n_x)
    P = E.square_implicit_problem(P0)

    @test E.API_sanity_checks(P).all_ok

    x = randn(n_x)
    y = fill(NaN, n_y)
    @inferred E.implicit_solve!(y, P, x)
    r = fill(NaN, n_y)
    # check solution via rootfinder
    @inferred E.implicit_residuals!(r, P, x, y)
    @test maximum(abs, r) ≤ 1e-10
    (; average_iterations) = E.get_statistics(P)
    @test isfinite(average_iterations) && average_iterations > 0

    @inferred E.calculate_∂y∂x(P, x, y)

    test_Enzyme_AD(P, P0)
end
