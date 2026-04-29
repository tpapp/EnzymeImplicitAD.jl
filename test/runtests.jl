using LinearAlgebra, Test, Enzyme, EnzymeTestUtils
import EnzymeImplicitAD as E

####
#### utilities for tests
####

"""
A linear test problem ``A⋅x + B⋅y = 0``, used for testing.
"""
struct MatrixProblem{TA<:AbstractMatrix,TB<:AbstractMatrix,TL}
    A::TA
    B::TB
    luB::TL
end

MatrixProblem(A::AbstractMatrix, B::AbstractMatrix) = MatrixProblem(A, B, lu(B))

MatrixProblem(n::Int) = MatrixProblem(randn(n, n), randn(n, n))

function E.implicit_solve!(y::AbstractVector{T}, P::MatrixProblem, x) where T
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

####
#### internals
####

@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n = 3
    (; A, B) = P = MatrixProblem(n)
    x = randn(n)
    y = randn(n)
    dy = similar(y)
    r = similar(x)
    dr = similar(x)

    J = similar(A)
    E.inplace_∂g∂y!(J, P, r, dr, x, y, dy)
    @test J ≈ B

    Jv = similar(r)
    v = randn(n)
    E.inplace_∂g∂x_v!(Jv, v, P, r, x, y)
    @test Jv ≈ A * v

    vJ = similar(r)
    v = randn(n)
    E.inplace_v_∂g∂x!(vJ, copy(v), P, r, x, y)
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
    @test dy ≈ -(B \ (A * dx))
end

@testset "reverse mode consistency test" begin
    n = 3
    (; A, B) = P = MatrixProblem(n)
    x = randn(n)
    dx = zeros(n)
    y = fill(NaN, n)
    dy = randn(n)
    expected_dx = (B \ A)' * (.-dy)
    autodiff(Reverse, E.implicit_solve!, Duplicated(y, dy), Const(P), Duplicated(x, dx))
    @test expected_dx ≈  dx
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

## NOTE add JET to the test environment, then uncomment
# using JET
# @testset "static analysis with JET.jl" begin
#     JET.test_package(EnzymeImplicitAD, target_modules=(EnzymeImplicitAD,))
# end

## NOTE add Aqua to the test environment, then uncomment
# @testset "QA with Aqua" begin
#     import Aqua
#     Aqua.test_all(EnzymeImplicitAD)
# end
