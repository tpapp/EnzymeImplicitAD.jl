include("setup.jl")

####
#### internals
####

@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n_x, n_y = 3, 4
    (; A, B) = P = LinearProblem(; n_y, n_x)
    x = randn(n_x)
    y = randn(n_y)
    dy = similar(y)
    b1 = similar(y)
    b2 = similar(y)
    b3 = similar(y)

    J = E._calculate_∂g∂y(P, x, y, b1, b2, b3)
    @test J ≈ B

    Jv = similar(y)
    v = randn(n_x)
    E._inplace_∂g∂x_v!(Jv, v, P, x, y, b1)
    @test Jv ≈ A * v

    vJ = similar(x)
    v = randn(n_y)
    E._inplace_v_∂g∂x!(vJ, copy(v), P, x, y, b1)
    @test vJ ≈ A' * v
end

@testset "linear problem AD test" begin
    P = LinearProblem(; n_x = 3, n_y = 4)
    test_Enzyme_AD(P, P)
end

@testset "implicit solver" begin
    n_x, n_y = 4, 5
    P0 = LinearProblem(; n_y, n_x, solver = false)
    P = E.square_implicit_problem(P0)

    @test E.API_sanity_checks(P).all_ok

    x = randn(n_x)
    y = fill(NaN, n_y)
    E.implicit_solve!(y, P, x)
    r = fill(NaN, n_y)
    # check solution via rootfinder
    @inferred E.implicit_residuals!(r, P, x, y)
    @test maximum(abs, r) ≤ 1e-10
    (; average_iterations) = E.get_statistics(P)
    @test isfinite(average_iterations) && average_iterations > 0

    test_Enzyme_AD(P, P0)
end

@testset "cached solver" begin
    n_x, n_y = 4, 5
    K = 10
    P0 = LinearProblem(; n_x, n_y)
    P = E.cache_implicit_problem(P0; min_size = K, max_size = 2 * K)

    @test E.API_sanity_checks(P).all_ok

    for _ in 1:(4*K)            # > max_size to test culling
        x = randn(n_x)
        y = fill(NaN, n_y)
        E.implicit_solve!(y, P, x)
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
        @test s.average_y_hit == (M - 1) / (M + 1) # one extra miss
        @test s.average_∂y∂x_hit == (M - 1) / (M + 1) # one extra miss
    end
end

@testset "static analysis with JET.jl" begin
    import JET
    JET.test_package(EnzymeImplicitAD, target_modules=(EnzymeImplicitAD,))
end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(EnzymeImplicitAD)
end
