@testset "cached solver" begin
    n_x, n_y = 4, 5
    K = 10
    P0 = LinearProblem(; n_x, n_y)
    P = E.cache_results(P0; min_size = K, max_size = 2 * K)

    @test E.sanity_checks(P).all_ok

    for _ in 1:(4*K)            # > max_size to test culling
        x = randn(n_x)
        y = fill(NaN, n_y)
        @inferred E.implicit_solve!(y, P, x)
        r = fill(NaN, n_y)
        # check solution via rootfinder
        E.implicit_residuals!(r, P, x, y)
        @test maximum(abs, r) ≤ 1e-10

        test_Enzyme_AD(P, P0)
    end
end

@testset "cached initial guesses" begin
    n_x, n_y = 4, 5
    P0 = LinearProblem(; n_x, n_y)
    x = randn(n_x)
    x2 = randn(n_x)
    @testset "!always_extrapolate" begin
        P = E.cache_results(P0; min_size = 5, max_size = 10, always_extrapolate = false)
        y = E.implicit_solve(P, x)

        # no partial calculation yet, we just get the other value
        y2 = only(E.get_initial_guesses(P, x2))
        @test y2 == y && y2 ≢ y     # equal, but not shared

        # calculating a partial, we get an exact solution
        E.calculate_∂y∂x(P, x, y)
        g2 = E.get_initial_guesses(P, x2)
        @test g2[2] == y && g2[2] ≢ y # as before
        r = zero(y)
        E.implicit_residuals!(r, P, x2, g2[1]) # better guess: exact
        @test sum(abs2, r) ≤ 1e-15
    end

    @testset "always_extrapolate" begin
        P = E.cache_results(P0; min_size = 5, max_size = 10, always_extrapolate = true)
        y = E.implicit_solve(P, x)
        g = E.get_initial_guesses(P, x2)
        @test g[2] == y && g[2] ≢ y
        r = zero(y)
        E.implicit_residuals!(r, P, x2, g[1]) # better guess: exact
        @test sum(abs2, r) ≤ 1e-15
    end
end

@testset "cache statistics" begin
    n_x = 4
    n_y = 5
    P0 = LinearProblem(; n_x, n_y)
    P = E.cache_results(P0)
    x = randn(n_x)
    y = fill(NaN, n_y)
    M = 10

    # just the value
    for _ in 1:M
        @inferred E.implicit_solve!(y, P, x)
    end
    s = E.get_statistics(P)
    @test s.average_y_hit == (M - 1) / M # all but one
    @test isnan(s.average_∂y∂x_hit)

    # just the derivative
    for _ in 1:M
        @inferred E.calculate_∂y∂x(P, x, y)
    end
    s = E.get_statistics(P)
    @test s.average_y_hit == (M - 1) / M    # same as before
    @test s.average_∂y∂x_hit == (M - 1) / M # all but one

    # another value
    x .+= 1
    @inferred E.calculate_∂y∂x(P, x, y)
    s = E.get_statistics(P)
    @test s.average_y_hit == (M - 1) / M          # same as before
    @test s.average_∂y∂x_hit == (M - 1) / (M + 1) # one extra miss
end
