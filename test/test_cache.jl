@testset "cached solver" begin
    n_x, n_y = 4, 5
    K = 10
    P0 = LinearProblem(; n_x, n_y)
    P = E.cache_implicit_problem(P0; min_size = K, max_size = 2 * K)

    @test E.API_sanity_checks(P).all_ok

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

    @testset "cache statistics" begin
        P0 = LinearProblem(; n_x, n_y)
        P = E.cache_implicit_problem(P0; min_size = K, max_size = 2 * K)
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
end
