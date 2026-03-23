using EnzymeImplicitAD, LinearAlgebra, Test, Enzyme, EnzymeTestUtils
using EnzymeImplicitAD: inplace_∂g∂x_v!, inplace_∂g∂y!, inplace_v_∂g∂x!

"""
Construct a linear test problem ``A⋅x + B⋅y = 0``.

Return the underlying matrices so that they can be used for analytically comparing
results.
"""
function make_test_problem(A, B)
    luB = lu(B)
    function g!(r::AbstractVector{T}, x, y) where T
        mul!(r, A, x)
        mul!(r, B, y, one(T), one(T))
        nothing
    end
    function f!(y::AbstractVector{T}, x) where T
        mul!(y, A, x, -one(T), zero(T))
        ldiv!(luB, y)
        nothing
    end
    ℐ = SquareImplicitFunction(f!, g!)
    (; f!, g!, A, B, ℐ)
end

"""
`make_test_problem` with random matrices of size `n`.
"""
rand_test_problem(n::Int) = make_test_problem(randn(n, n), randn(n, n))

@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n = 3
    (; f!, g!, A, B) = rand_test_problem(n)
    x = randn(n)
    y = randn(n)
    dy = similar(y)
    r = similar(x)
    dr = similar(x)

    J = similar(A)
    inplace_∂g∂y!(J, g!, r, dr, x, y, dy)
    @test J ≈ B

    Jv = similar(r)
    v = randn(n)
    inplace_∂g∂x_v!(Jv, v, g!, r, x, y)
    @test Jv ≈ A * v

    vJ = similar(r)
    v = randn(n)
    inplace_v_∂g∂x!(vJ, copy(v), g!, r, x, y)
    @test vJ ≈ A' * v
end

@testset "forward mode consistency test" begin
    n = 3
    (; A, B, ℐ) = rand_test_problem(n)
    x = randn(n)
    dx = randn(n)
    y = zeros(n)
    dy = zeros(n)
    autodiff(Forward, Const(ℐ), Duplicated(y, dy), Duplicated(x, dx))
    @test dy ≈ -(B \ (A * dx))
end

@testset "reverse mode consistency test" begin
    n = 3
    (; A, B, ℐ) = rand_test_problem(n)
    x = randn(n)
    dx = zeros(n)
    y = fill(NaN, n)
    dy = randn(n)
    expected_dx = (B \ A)' * (.-dy)
    autodiff(Reverse, Const(ℐ), Duplicated(y, dy), Duplicated(x, dx))
    @test expected_dx ≈  dx
end

@testset "testing with EnzymeTestUtils" begin
    n = 3
    (; ℐ) = rand_test_problem(n)
    x = randn(n)
    y = similar(x)
    @testset "test_forward" begin
        @testset for Tx in (Const, Duplicated,), Ty in (Const, Duplicated)
            test_forward(ℐ, Const, (y, Tx), (x, Ty))
        end
        @testset for Tx in (Const, Duplicated,), Ty in (Const, Duplicated)
            test_forward(ℐ, Const, (y, Tx), (x, Ty))
        end
    end
    ℐ = make_test_problem(I(3), I(3)).ℐ
    @testset "test_reverse" begin
        test_reverse(ℐ, Const, (y, Duplicated), (x, Duplicated))
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
