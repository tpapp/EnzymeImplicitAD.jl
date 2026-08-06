#####
##### the generic API
#####

public get_dimensions, get_preferred_eltype, is_square, initial_guess!, implicit_solve!,
    implicit_residuals!, task_local_buffers, calculate_∂y∂x, calculate_pushforward!,
    accumulate_pullback!, get_statistics

"""
$(FUNCTIONNAME)(implicit_problem) → (; n_x, n_y, n_r)

Return the dimensions of the problem.
"""
function get_dimensions end

"""
$(SIGNATURES) → T

Return the preferred element type for a problem. This is used for buffers and interim
quantities, and should allow for enough precision even with input/output arrays that
have less.

The default is `Float64`.
"""
get_preferred_eltype(implicit_problem) = Float64

"""
$(SIGNATURES)

Return `true` iff the problem is square (`y`, `r` have the same dimensions).
"""
function is_square(implicit_problem)
    (; n_y, n_r) = get_dimensions(implicit_problem)
    n_y == n_r
end

"""
$(SIGNATURES) → nothing

Provide an initial guess for the problem given `x`, into `y`.

Return `nothing`.

Caller can assume that the dimensions are correct.
"""
initial_guess!(y, problem, x) = (fill!(y, zero(eltype(y))); nothing)

"""
$(FUNCTIONNAME)(implicit_problem) → solver

Return the solver for use in [`implicit_solve_with_solver!`](@ref).
"""
function get_solver end

"""
$(FUNCTIONNAME)((y, implicit_problem, solver, x) → nothing

Solve with `implicit_problem` at `x` with `solver` called.

`y` should contain a valid initial guess when called.

The result is put in `y`.
"""
function implicit_solve_with_solver! end

"""
$(SIGNATURES) → nothing

Solve the implicit problem ``g(x, y(x)) = 0`` at `x`, overwriting `y` with ``y(x)`` result.

Return `nothing`. See [`implicit_residuals!`](@ref), which implements ``g`` above.

!!! NOTE
    Don't specialize this method, rather [`initial_guess!`](@ref),
    [`impicit_solve_with_solver!`](@ref) and [`get_solver`](@ref).
```
"""
function implicit_solve!(y, implicit_problem, x)
    solver = get_solver(implicit_problem)
    initial_guess!(y, implicit_problem, x)
    implicit_solve_with_solver!(y, implicit_problem, solver, x)
end

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

Should be a concrete type that depends only on `implicit_problem`, not `x` or `y`.

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
    @assert is_square(implicit_problem)
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
    @assert is_square(implicit_problem)
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

"""
$(SIGNATURES) → statistics::NamedTuple

Return various statistics that are accumulated during calls, that may help the user
evaluate and tune algorithms.

!!! implementation note
    Wrapper types should merge statistics of the parent in most cases, checking that
    they don't overwrite. See [`merge_disjoint`](@ref).
"""
get_statistics(problem) = (;)
