@testset "∂Y∂X pretty printing" begin
    repr(E.calculate_∂y∂x(LinearProblem(; n_x = 3, n_y = 4), randn(3), randn(4))) ==
        "∂Y∂X(« 4 × 4 »)"
end

@testset "linear problem AD test" begin
    P = LinearProblem(; n_x = 3, n_y = 4)
    test_Enzyme_AD(P, P)
end

"A trivial test problem that fails when x < 0."
struct MayFail end

E.get_dimensions(::MayFail) = (; n_x = 1, n_y = 1, n_r = 1)

function E.implicit_solve_with_initial_guesses!(y, ::MayFail, x; initial_guesses)
    y[1] = x[1]
    x[1] ≥ 0                    # flag
end

E.implicit_residuals!(r, ::MayFail, x, y) = (r[1] = x[1] - y[1]; nothing)

@testset "failure" begin
    f = MayFail()
    c = E.API_sanity_checks(f)
    x = [-1.0]
    y = [NaN]
    dx = copy(x)
    dy = copy(y)
    # failure
    @test !(@inferred(E.implicit_solve!(y, f, x)))
    @test_throws E.NonSolutionAD autodiff(ForwardWithPrimal, E.implicit_solve!,
                                          Duplicated(y, dy), Const(f), Duplicated(x, dx))
    @test_throws E.NonSolutionAD autodiff(ReverseWithPrimal, E.implicit_solve!, Duplicated(y, dy),
                                          Const(f), Duplicated(x, dx))
    # success
    x[1] = 1.0
    @test @inferred(E.implicit_solve!(y, f, x))
    @test @inferred(autodiff(ForwardWithPrimal, E.implicit_solve!,
                             Duplicated(y, dy), Const(f), Duplicated(x, dx))) == (true,)
    @test @inferred(autodiff(ReverseWithPrimal, E.implicit_solve!, Duplicated(y, dy),
                             Const(f), Duplicated(x, dx))) == ((nothing, nothing, nothing), true)
end
