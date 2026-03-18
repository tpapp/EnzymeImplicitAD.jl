using EnzymeImplicitAD, LinearAlgebra, Test, Enzyme
using EnzymeImplicitAD: inplace_∂g∂x_v!, inplace_∂g∂y!

function test_problem(n::Int)
    A = randn(n, n)
    B = randn(n, n)
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

@testset "∂g∂y and ∂g∂x_v extraction" begin
    n = 3
    (; f!, g!, A, B) = test_problem(n)
    J = similar(A)
    x = randn(n)
    y = randn(n)
    dy = similar(y)
    r = similar(x)
    dr = similar(x)
    inplace_∂g∂y!(J, g!, r, dr, x, y, dy)
    @test J ≈ B
    Jv = similar(r)
    v = randn(n)
    inplace_∂g∂x_v!(Jv, g!, r, x, v, y)
    @test Jv ≈ A * v
end

@testset "forward mode consistency test" begin
    n = 3
    (; A, B, ℐ) = test_problem(n)
    x = randn(n)
    dx = randn(n)
    y = zeros(n)
    dy = zeros(n)
    autodiff(Forward, Const(ℐ), Duplicated(y, dy), Duplicated(x, dx))
    @test dy ≈ -(B \ (A * dx))
end

# write tests here

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
