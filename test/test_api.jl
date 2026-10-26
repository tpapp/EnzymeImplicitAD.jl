@testset "∂Y∂X pretty printing" begin
    repr(E.calculate_∂y∂x(LinearProblem(; n_x = 3, n_y = 4), randn(3), randn(4))) ==
        "∂Y∂X(« 4 × 4 »)"
end

@testset "linear problem AD test" begin
    P = LinearProblem(; n_x = 3, n_y = 4)
    test_Enzyme_AD(P, P)
end

@testset "failure" begin
    f = MayFail()
    c = E.API_sanity_checks(f)
    x = [-1.0]
    y = [NaN]
    dx = copy(x)
    dy = copy(y)
    # failure
    @test !(@inferred(E.implicit_solve!(y, f, x)))
    @inferred autodiff(ForwardWithPrimal, E.implicit_solve!, Duplicated(y, dy), Const(f),
                       Duplicated(x, dx))
    @test all(isnan, dy)
    dy .= 1.0
    @inferred autodiff(ReverseWithPrimal, E.implicit_solve!, Duplicated(y, dy), Const(f),
                       Duplicated(x, dx))
    @test all(isnan, dx)
    # success
    x[1] = 1.0
    @test @inferred(E.implicit_solve!(y, f, x))
    @test @inferred(autodiff(ForwardWithPrimal, E.implicit_solve!,
                             Duplicated(y, dy), Const(f), Duplicated(x, dx))) == (true,)
    @test @inferred(autodiff(ReverseWithPrimal, E.implicit_solve!, Duplicated(y, dy),
                             Const(f), Duplicated(x, dx))) == ((nothing, nothing, nothing), true)
end

@testset "chained wrappers innermost parent" begin
    P0 = LinearProblem(; n_x = 3, n_y = 4)
    @test E.innermost_parent(P0) ≡ P0
    P1 = (E.keep_failures ∘ E.cache_results ∘ E.preallocate_buffers)(P0)
    @test E.innermost_parent(P1) ≡ P0
end
