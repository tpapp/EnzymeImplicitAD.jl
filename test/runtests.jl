using EnzymeImplicitAD
import EnzymeImplicitAD as E
using LinearAlgebra, Test, Enzyme, EnzymeTestUtils, FiniteDifferences

####
#### utilities for tests
####

###
### functional interface
###

function implicit_solve(implicit_problem, x)
    T = E.get_preferred_eltype(implicit_problem)
    (; n_y) = E.get_dimensions(implicit_problem)
    y = Vector{T}(undef, n_y)
    E.implicit_solve!(y, implicit_problem, x)
    y
end

###
### FDM pushforward and pullback
###

const FDM = central_fdm(5, 1)

fdm_pushforward(P, x, dx; fdm = FDM) = jvp(fdm, Base.Fix1(implicit_solve, P), (x, dx))

fdm_pullback(P, x, dy; fdm = FDM) = j′vp(fdm, Base.Fix1(implicit_solve, P), dy, x)[1]

###
### linear test problem
###

"""
A linear test problem ``A⋅x + B⋅y = 0``, used for testing.

The `solver::Bool` argument (`S` parameter) determines whether [`implicit_solve!`](@ref)
is implemented.
"""
struct MatrixProblem{S,TA<:AbstractMatrix,TB<:AbstractMatrix,TL}
    A::TA
    B::TB
    luB::TL
    function MatrixProblem(A::AbstractMatrix, B::AbstractMatrix; solver::Bool = true)
        luB = lu(B)
        new{solver,typeof(A),typeof(B),typeof(luB)}(A, B, lu(B))
    end
end

"""
Make a random (square, `n_r = n_y`) matrix problem with the given dimensions.

When `solver = true` (the default), a solver method is implemented, otherwise it errors.
"""
function MatrixProblem(; n_y::Int, n_x::Int = n_y, solver::Bool = true)
    @assert n_x > 0
    @assert n_y > 0
    MatrixProblem(randn(n_y, n_x), randn(n_y, n_y); solver)
end

function E.get_dimensions(P::MatrixProblem)
    n_y, n_x = size(P.A)
    (; n_x, n_y, n_r = n_y)
end

E.get_preferred_eltype(P::MatrixProblem) = eltype(P.A)

function E.implicit_solve!(y::AbstractVector{T}, P::MatrixProblem{true}, x) where T
    (; A, luB) = P
    mul!(y, A, x, -one(T), zero(T))
    ldiv!(luB, y)
    nothing
end

function E.implicit_residuals!(r::AbstractVector{T}, P::MatrixProblem, x, y) where T
    (; A, B) = P
    mul!(r, A, x)
    mul!(r, B, y, one(T), one(T))
    nothing
end

analytical_pushforward(P::MatrixProblem, dx) = -(P.luB \ (P.A * dx))

analytical_pullback(P::MatrixProblem, dy) = (P.luB \ P.A)' * (.-dy)

@test E.API_sanity_checks(MatrixProblem(; n_x = 3, n_y = 4)).all_ok

####
#### internals
####

@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n_x, n_y = 3, 4
    (; A, B) = P = MatrixProblem(; n_y, n_x)
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

@testset "forward mode consistency test" begin
    n_x, n_y = 3, 4
    P = MatrixProblem(; n_x, n_y)
    x = randn(n_x)
    dx = randn(n_x)
    y = zeros(n_y)
    dy = zeros(n_y)
    autodiff(Forward, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
    @test dy ≈ analytical_pushforward(P, dx)
end

@testset "reverse mode consistency test" begin
    n_x, n_y = 3, 2
    P = MatrixProblem(; n_y, n_x)
    x = randn(n_x)
    dx = zeros(n_x)
    y = fill(NaN, n_y)
    dy = randn(n_y)
    expected_dx = analytical_pullback(P, dy)
    autodiff(Reverse, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
    @test expected_dx ≈ dx
end

@testset "testing with EnzymeTestUtils" begin
    n_x, n_y = 3, 4
    P = MatrixProblem(; n_x, n_y)
    x = randn(n_x)
    y = zeros(n_y)
    @testset "test_forward" begin
        @testset for Tx in (Const, Duplicated,), Ty in (Const, Duplicated)
            test_forward(E.implicit_solve!, Const, (y, Ty), P, (x, Tx))
        end
    end
    @testset "test_reverse" begin
        test_reverse(E.implicit_solve!, Const, (y, Duplicated), P, (x, Duplicated))
    end
end

@testset "implicit solver" begin
    n_x, n_y = 4, 5
    P0 = MatrixProblem(; n_y, n_x, solver = false)
    P = E.square_implicit_problem(P0)

    @test E.API_sanity_checks(P).all_ok

    x = randn(n_x)
    y = fill(NaN, n_y)
    E.implicit_solve!(y, P, x)
    r = fill(NaN, n_y)
    # check solution via rootfinder
    E.implicit_residuals!(r, P, x, y)
    @test maximum(abs, r) ≤ 1e-10

    @testset "test_forward" begin
        @testset for Tx in (Const, Duplicated,), Ty in (Const, Duplicated)
            test_forward(E.implicit_solve!, Const, (y, Ty), P, (x, Tx))
        end
    end
    @testset "test_reverse" begin
        test_reverse(E.implicit_solve!, Const, (y, Duplicated), P, (x, Duplicated))
    end
end

@testset "cached solver" begin
    n_x, n_y = 4, 5
    K = 10
    P0 = MatrixProblem(; n_x, n_y)
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

        # pushforward
        dx = randn(n_x)
        dy = fill(NaN, n_y)
        autodiff(Forward, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
        @test dy ≈ fdm_pushforward(P, x, dx) atol = 1e-8

        # pullback
        dy0 = randn(n_y)
        dy = copy(dy0)
        dx0 = randn(n_x)
        dx = copy(dx0)
        autodiff(Reverse, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
        @test dx ≈ (fdm_pullback(P, x, dy0) .+ dx0) atol = 1e-8
    end

    # NOTE cf https://github.com/EnzymeAD/Enzyme.jl/issues/3123
    # disabled for now, use manual testing above
    # @testset "test_forward" begin
    #     @testset for Tx in (Const, Duplicated,), Ty in (Const, Duplicated)
    #         test_forward(E.implicit_solve!, Const, (y, Ty), P, (x, Tx))
    #     end
    # end
    # @testset "test_reverse" begin
    #     test_reverse(E.implicit_solve!, Const, (y, Duplicated), P, (x, Duplicated))
    # end
end

## NOTE add JET to the test environment, then uncomment
# using JET
# @testset "static analysis with JET.jl" begin
#     JET.test_package(EnzymeImplicitAD, target_modules=(EnzymeImplicitAD,))
# end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(EnzymeImplicitAD)
end
