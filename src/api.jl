#####
##### the generic API
#####

public
    # introspection
    get_dimensions,
    ZeroResiduals,
    get_solution_concept,
    get_preferred_eltype,
    get_statistics,
    # solution
    get_initial_guesses,
    implicit_solve!,
    implicit_residuals!,
    # helpers
    task_local_buffers,
    calculate_∂y∂x,
    calculate_pushforward!,
    accumulate_pullback!

####
#### introspection
####

"""
$(FUNCTIONNAME)(implicit_problem) → (; n_x, n_y, n_r)

Return the dimensions of the problem.
"""
function get_dimensions end

"""
Solution concept: for a given `x`, `y` is the solution which makes residuals `r` zero in
`implicit_residuals!(r, problem, x, y)`.
"""
struct ZeroResiduals end

"""
$(FUNCTIONNAME)(implicit_problem) → solution_concept

Return the solution concept. The default is [`ZeroResiduals`](@ref).
"""
get_solution_concept(problem) = ZeroResiduals()

"""
$(SIGNATURES) → T

Return the preferred element type for a problem. This is used for buffers and interim
quantities, and should allow for enough precision even with input/output arrays that
have less.

The default is `Float64`.
"""
get_preferred_eltype(implicit_problem) = Float64

"""
$(SIGNATURES) → statistics::NamedTuple

Return various statistics that are accumulated during calls, that may help the user
evaluate and tune algorithms.

!!! implementation note
    Wrapper types should merge statistics of the parent in most cases, checking that
    they don't overwrite. See [`merge_disjoint`](@ref).
"""
get_statistics(problem) = (;)

####
#### solver
####

"""
$(SIGNATURES) → AbstractVector{<:AbstractVector{eltype(x)}}

Provide initial guess(es) for the problem given `x`, as a vector of vectors. May be
empty. The most useful initial guesses should come first.

Caller can assume that the dimensions are correct.
"""
get_initial_guesses(problem, x) = Vector{typeof(x)}()

"""
$(FUNCTIONNAME)((y, implicit_problem, x; initial_guesses) → nothing

Solve for `y` with `implicit_problem` at `x`. `initial_guesses` is a vector of
initial guesses for `y` (may be empty).

The result is put in `y`.

Methods are implemented *outside* this package.
"""
function implicit_solve_with_initial_guesses! end

"""
$(SIGNATURES) → nothing

Solve the implicit problem ``g(x, y(x)) = 0`` at `x`, overwriting `y` with ``y(x)`` result.

Return `nothing`. See [`implicit_residuals!`](@ref), which implements ``g`` above.

!!! NOTE
    Don't specialize this method, rather [`get_initial_guesses`](@ref),
    [`impicit_solve_with_initial_guesses!`](@ref).
```
"""
function implicit_solve!(y, implicit_problem, x)
    initial_guesses = get_initial_guesses(implicit_problem, x)
    implicit_solve_with_initial_guesses!(y, implicit_problem, x; initial_guesses)
end

####
#### residuals
####

"""
$(FUNCTIONNAME)(r, implicit_problem, x, y) → nothing

Calculate the implicit residuals ``r = g(x, y)``, overwriting `r`.

Return `nothing`.

It is assumed that after
```julia
implicit_solve!(y, implicit_problem, x)
$(FUNCTIONNAME)(r, implicit_problem, x, y)
```
the residuals `r` are “approximately” zero, but this is not checked.
"""
function implicit_residuals! end

"""
$(SIGNATURES) → (; buffer_y1, buffer_y2, buffer_y3)

Return an object which contains the following buffers, which are accessible as
properties. Each is a vector, with lengths consistent with the corresponding dimension
in [`get_dimensions`](@ref).

- `buffer_r`, `buffer_r2`: has length `n_r`
- `buffer_x`: has length `n_x`
- `buffer_y`: has length `n_y`

The fallback method reallocates these for each use, implementations can provide shared
buffers but they are guaranteed to be task-local.

The element type of buffers should be consistent with [`get_preferred_eltype`](@ref).

$(BUFFER_DOCS)
"""
function task_local_buffers(implicit_problem)
    _make_buffers(get_preferred_eltype(implicit_problem); get_dimensions(implicit_problem)...)
end

@concrete struct ∂Y∂X
    ∂g∂y_factor
end

function Base.show(io::IO, ∂y∂x::∂Y∂X)
    n_x, n_y = size(∂y∂x.∂g∂y_factor)
    print(io, "∂Y∂X(« $(n_x) × $(n_y) »)")
end

"""
$(SIGNATURES)

The return type of [`calculate_∂y∂x`](@ref).

Should be a concrete type that depends only on the type of `implicit_problem`, not `x`
or `y`.

Used-defined methods should ensure consistency.
"""
function get_∂y∂x_type(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    L = typeof(lu!(ones(T::Type, 1, 1))) # assumption: lu! is type stable, size does not matter
    ∂Y∂X{L}
end

"""
$(SIGNATURES)

Return an object `∂y∂x` that acts like a Jacobian matrix when pre- or post-multiplied by
a conformable vector, via the methods [`calculate_pushforward!`](@ref) and
[`accumulate_pullback!`](@ref).

The return type should depend only on `implicit_problem`, and should be consistent with
[`get_∂y∂x_type`]((@ref).

The implementation is free to ignore `y`, eg if it can obtain a solution from `x`.
"""
function calculate_∂y∂x(implicit_problem, x, y)
    (; buffer_y, buffer_r, buffer_r2) = task_local_buffers(implicit_problem)
    ∂g∂y = _calculate_∂g∂y(implicit_problem, x, y, buffer_y, buffer_r, buffer_r2)
    ∂Y∂X(lu!(∂g∂y))
end

"""
$(SIGNATURES)

Calculate the pushforward `dy = ∂y∂x ⋅ dx` into `dy`.

A fallback is provided using Enzyme, but an `implicit_problem` can define its own method.
"""
function calculate_pushforward!(dy, implicit_problem, x, y, ∂y∂x::∂Y∂X, dx)
    @argcheck get_solution_concept(implicit_problem) ≡ ZeroResiduals()
    (; buffer_r) = task_local_buffers(implicit_problem)
    _inplace_∂g∂x_v!(dy, dx, implicit_problem, x, y, buffer_r)
    ldiv!(∂y∂x.∂g∂y_factor, dy)
    dy .*= -1
    nothing
end

"""
$(SIGNATURES)

Accumulate the pullback `dy ⋅ ∂y∂x` into `dx`.

A default is implemented using Enzyme, but an `implicit_problem` can define its own method.
"""
function accumulate_pullback!(dx, implicit_problem, x, y, ∂y∂x::∂Y∂X, dy)
    @argcheck get_solution_concept(implicit_problem) ≡ ZeroResiduals()
    (; ∂g∂y_factor) = ∂y∂x
    # math:
    #     dy ⋅ ∂y/∂x = - (dy' / ∂g/∂y) ⋅ ∂g/∂x
    (; buffer_x, buffer_y, buffer_r) = task_local_buffers(implicit_problem)
    buffer_y .= dy                # buffer_y1 == dy
    rdiv!(buffer_y', ∂g∂y_factor) # a == dy' / ∂g∂y
    _inplace_v_∂g∂x!(buffer_x, buffer_y, implicit_problem, x, y,
                     buffer_r)  # b = (dy' / ∂g/∂y) ⋅ ∂g/∂x
    dx .-= buffer_x
    nothing
end
