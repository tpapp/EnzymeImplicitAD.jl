using LinearAlgebra, Test, Enzyme, EnzymeTestUtils
using EnzymeImplicitAD
import EnzymeImplicitAD as E

####
#### utilities for tests
####

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

MatrixProblem(n::Int; solver::Bool = true) = MatrixProblem(randn(n, n), randn(n, n); solver)

function E.get_dimensions(P::MatrixProblem)
    n = size(P.A, 1)
    (; n_x = n, n_y = n, n_r = n)
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

@test E.API_sanity_checks(MatrixProblem(3)).all_ok

####
#### internals
####

@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n = 3
    (; A, B) = P = MatrixProblem(n)
    x = randn(n)
    y = randn(n)
    dy = similar(y)
    b1 = similar(y)
    b2 = similar(y)
    b3 = similar(y)

    J = E._calculate_∂g∂y(P, x, y, b1, b2, b3)
    @test J ≈ B

    Jv = similar(y)
    v = randn(n)
    E._inplace_∂g∂x_v!(Jv, v, P, x, y, b1)
    @test Jv ≈ A * v

    vJ = similar(y)
    v = randn(n)
    E._inplace_v_∂g∂x!(vJ, copy(v), P, x, y, b1)
    @test vJ ≈ A' * v
end

@testset "forward mode consistency test" begin
    n = 3
    (; A, B) = P = MatrixProblem(n)
    x = randn(n)
    dx = randn(n)
    y = zeros(n)
    dy = zeros(n)
    autodiff(Forward, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
    @test dy ≈ analytical_pushforward(P, dx)
end

@testset "reverse mode consistency test" begin
    n = 3
    (; A, B) = P = MatrixProblem(n)
    x = randn(n)
    dx = zeros(n)
    y = fill(NaN, n)
    dy = randn(n)
    expected_dx = analytical_pullback(P, dy)
    autodiff(Reverse, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
    @test expected_dx ≈ dx
end

@testset "testing with EnzymeTestUtils" begin
    n = 3
    P = MatrixProblem(n)
    x = randn(n)
    y = similar(x)
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
    n = 4
    P0 = MatrixProblem(n; solver = false)
    P = E.square_implicit_problem(P0)
    x = randn(n)
    y = fill(NaN, n)
    E.implicit_solve!(y, P, x)
    r = fill(NaN, n)
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

## NOTE add JET to the test environment, then uncomment
# using JET
# @testset "static analysis with JET.jl" begin
#     JET.test_package(EnzymeImplicitAD, target_modules=(EnzymeImplicitAD,))
# end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(EnzymeImplicitAD)
end
