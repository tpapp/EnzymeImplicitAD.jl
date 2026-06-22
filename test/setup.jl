#####
##### test framework setup and utility functions
#####

using EnzymeImplicitAD
import EnzymeImplicitAD as E
using LinearAlgebra, Test, Enzyme

####
#### linear problem
####

"""
A linear test problem ``A⋅x + B⋅y = 0``, used for testing.

The `solver::Bool` argument (`S` parameter) determines whether [`implicit_solve!`](@ref)
is implemented.
"""
struct LinearProblem{S,TA<:AbstractMatrix,TB<:AbstractMatrix,TL}
    A::TA
    B::TB
    luB::TL
    function LinearProblem(A::AbstractMatrix, B::AbstractMatrix; solver::Bool = true)
        luB = lu(B)
        new{solver,typeof(A),typeof(B),typeof(luB)}(A, B, lu(B))
    end
end

function Base.show(io::IO, problem::LinearProblem)
    (; n_x, n_y) = E.get_dimensions(problem)
    print(io, "« $(n_y) × $(n_x) linear problem »")
end

"""
Make a random (square, `n_r = n_y`) matrix problem with the given dimensions.

When `solver = true` (the default), a solver method is implemented, otherwise it errors.
"""
function LinearProblem(; n_y::Int, n_x::Int = n_y, solver::Bool = true)
    @assert n_x > 0
    @assert n_y > 0
    LinearProblem(randn(n_y, n_x), randn(n_y, n_y); solver)
end

function E.get_dimensions(P::LinearProblem)
    n_y, n_x = size(P.A)
    (; n_x, n_y, n_r = n_y)
end

function E.implicit_solve!(y::AbstractVector{T}, P::LinearProblem{true}, x) where T
    (; A, luB) = P
    mul!(y, A, x, -one(T), zero(T))
    ldiv!(luB, y)
    nothing
end

function E.implicit_residuals!(r::AbstractVector{T}, P::LinearProblem, x, y) where T
    (; A, B) = P
    mul!(r, A, x)
    mul!(r, B, y, one(T), one(T))
    nothing
end

analytical_pushforward(P::LinearProblem, dx) = -(P.luB \ (P.A * dx))

analytical_pullback(P::LinearProblem, dy) = (P.luB \ P.A)' * (.-dy)


"""
Test forward and reverse AD with Enzyme for `implicit_problem`.

`analytical_problem` should yield derivatives to compare to, using `analytical_pullback`
and `analytical_pushforward`.
"""
function test_Enzyme_AD(implicit_problem, analytical_problem;
                        testset_name = "AD tests for $(repr(implicit_problem))",
                        atol = 1e-6)
    (; n_x, n_y, n_r) = E.get_dimensions(implicit_problem)
    @testset "$(testset_name)" begin
        r = zeros(n_r)
        @testset "forward" begin
            x = randn(n_x)
            y = fill(NaN, n_y)
            dx = randn(n_x)
            dy = fill(NaN, n_y)
            expected_dy = analytical_pushforward(analytical_problem, dx)
            autodiff(Forward, E.implicit_solve!, Duplicated(y, dy), Const(implicit_problem),
                     Duplicated(x, dx))
            @test dy ≈ expected_dy atol = atol
            E.implicit_residuals!(r, implicit_problem, x, y)
            @test sum(abs2, r) ≤ atol
        end
        @testset "reverse" begin
            x = randn(n_x)
            y = randn(n_y)
            dx = randn(n_x)
            dy = randn(n_y)
            expected_dx = dx .+ analytical_pullback(analytical_problem, dy)
            autodiff(Reverse, E.implicit_solve!, Duplicated(y, dy), Const(implicit_problem),
                     Duplicated(x, dx))
            @test dx ≈ expected_dx atol = atol
            E.implicit_residuals!(r, implicit_problem, x, y)
            @test sum(abs2, r) ≤ atol
        end
    end
end

####
#### test benchmarks
####

"Solver and derivatives fail with a given probability."
struct SometimesFails{P}
    inner_problem::P
    failure_probability::Float64
end

E.get_dimensions(s::SometimesFails) = E.get_dimensions(s.inner_problem)

E.get_preferred_eltype(s::SometimesFails) = E.get_preferred_eltype(s.inner_problem)

function E.implicit_solve!(y, s::SometimesFails, x)
    rand() < s.failure_probability && throw(DomainError(copy(x), "I don't like this particular x"))
    E.implicit_solve!(y, s.inner_problem, x)
end

function E.implicit_residuals!(r, s::SometimesFails, x, y)
    E.implicit_residuals!(r, s.inner_problem, x, y)
end

function E.calculate_∂y∂x(s::SometimesFails, x, y)
    rand() < s.failure_probability && throw(DomainError(copy(x), "I don't like this particular x"))
    E.calculate_∂y∂x(s.inner_problem, x, y)
end
