using EnzymeImplicitAD
using LinearAlgebra, Test
using EnzymeImplicitAD: inplace_∂r∂y!

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
    F = SquareImplicitFunction(f!, g!)
    (; f!, g!, A, B, F)
end

n = 3
(; f!, g!, A, B, F) = test_problem(n)
J = similar(A)
x = randn(n)
y = randn(n)
dy = similar(y)
r = similar(x)
dr = similar(x)
inplace_∂r∂y!(J, g!, r, dr, x, y, dy)
@test J ≈ B

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
