#####
##### the generic API
#####

public
    # accessors
    get_dimensions,
    ZeroResiduals,
    get_solution_concept,
    get_preferred_eltype,
    get_statistics,
    get_buffers,
    independent_copy,
    # solution
    get_initial_guesses,
    implicit_solve_with_initial_guesses!,
    implicit_solve!,
    implicit_residuals!,
    # helpers
    calculate_∂y∂x,
    calculate_pushforward!,
    accumulate_pullback!

####
#### introspection
####

"""
`$(FUNCTIONNAME)(implicit_problem)` → `(; n_x, n_y, n_r)`

Return the dimensions of the problem.
"""
function get_dimensions end

"""
Solution concept: for a given `x`, `y` is the solution which makes residuals `r` zero in
`implicit_residuals!(r, problem, x, y)`.
"""
struct ZeroResiduals end

"""
`$(FUNCTIONNAME)(implicit_problem)` → `solution_concept`

Return the solution concept. The default is [`ZeroResiduals`](@ref).
"""
get_solution_concept(problem) = ZeroResiduals()

"""
`$(SIGNATURES)` → `T`

Return the preferred element type for a problem. This is used for buffers and interim
quantities, and should allow for enough precision even with input/output arrays that
have less.

The default is `Float64`.
"""
get_preferred_eltype(implicit_problem) = Float64

"""
`$(SIGNATURES)` → `statistics::NamedTuple`

Return various statistics that are accumulated during calls, that may help the user
evaluate and tune algorithms.

!!! implementation note
    Wrapper types should merge statistics of the parent in most cases, checking that
    they don't overwrite. See [`merge_disjoint`](@ref).
"""
get_statistics(problem) = (;)

"""
$(SIGNATURES)

Make a copy of problem that functionally equivalent but independent, ie safe for using
in another thread. Specifically, this should ensure that all the functions in the API of
this package are safe to use concurrently on independent copies.

The generic method makes an effort to just return recursively immutable values as is,
but it may just be too narrow. If it fails, define a method for the type.
"""
function independent_copy(x::T) where T
    if _is_recursively_immutable(x)
        x
    else
        error("Define a method for `independent_copy` for types $T.")
    end
end

####
#### solver
####

"""
`$(SIGNATURES)` → `AbstractVector{<:AbstractVector{eltype(x)}}`

Provide initial guess(es) for the problem given `x`, as a vector of vectors. May be
empty. The most useful initial guesses should come first.

Caller can assume that the dimensions are correct.
"""
get_initial_guesses(problem, x) = Vector{typeof(x)}()

"""
`$(FUNCTIONNAME)(y, implicit_problem, x; initial_guesses)` → `success::Bool`

Solve for `y` with `implicit_problem` at `x`. `initial_guesses` is a vector of
initial guesses for `y` (may be empty).

When `success`, the result is put in `y`. When `!success`, the contents of `y` are
undefined.

Methods are implemented *outside* this package.
"""
function implicit_solve_with_initial_guesses! end

"""
`$(SIGNATURES)` → `success::Bool`

Solve the implicit problem ``g(x, y(x)) = 0`` at `x`, overwriting `y` with ``y(x)`` result.

When `success`, the result is put in `y`. When `!success`, the contents of `y` (and,
consequently, the pullback/pushforward in AD) are undefined. Implementations are
encouraged, but not required, to fill the latter with `NaN`s.

See [`implicit_residuals!`](@ref), which implements ``g`` above.

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
`$(FUNCTIONNAME)(r, implicit_problem, x, y)` → `nothing`

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
`$(SIGNATURES)` → `(; x, y, dy, r, dr)`

Return an object which contains the following buffers, which are accessible as
properties. Each is a vector, with lengths consistent with the corresponding dimension
in [`get_dimensions`](@ref).

- `buffer_r` is a vector of length `n_r`
- `buffer_dr` is a tuple of $(AD_CHUNKS) vectors, each of length `n_r`
- `buffer_x` is a vector of length `n_x`
- `buffer_y` is a vector of length `n_y`
- `buffer_dy` is a tuple of $(AD_CHUNKS) vectors, each of length `n_y`

The element type of buffers should be consistent with [`get_preferred_eltype`](@ref).

The hardcoded constant $(AD_CHUNKS) is available as [`AD_CHUNKS`](@ref).

The fallback method reallocates these for each use, implementations can provide reusable
buffers. However, the buffers should not be used concurrently, see [`independent_copy`](@ref).
"""
function get_buffers(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    (; n_r, n_y, n_x) = get_dimensions(implicit_problem)
    _v(n) = Vector{T}(undef, n)
    _vc(n) = ntuple(_ -> _v(n), Val(AD_CHUNKS))
    (; x = _v(n_x), y = _v(n_y), r = _v(n_r), dy = _vc(n_y), dr = _vc(n_r))
end

struct ∂Y∂X{T}
    ∂g∂y_factor::T
end

function Base.show(io::IO, ∂y∂x::∂Y∂X)
    n_x, n_y = size(∂y∂x.∂g∂y_factor)
    print(io, "∂Y∂X(« $(n_x) × $(n_y) »)")
end

"""
`$(SIGNATURES) → T`

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
`$(SIGNATURES)` → `∂y∂x`

Return an object `∂y∂x` that acts like a Jacobian matrix when pre- or post-multiplied by
a conformable vector, via the methods [`calculate_pushforward!`](@ref) and
[`accumulate_pullback!`](@ref).

The return type should depend only on `implicit_problem`, and should be consistent with
[`get_∂y∂x_type`]((@ref).

The implementation is free to ignore `y`, eg if it can obtain a solution from `x`.
However, this function should only be called when there is a solution (`success ==
true`), otherwise consequences are undefined.
"""
function calculate_∂y∂x(implicit_problem, x, y)
    buffers = get_buffers(implicit_problem)
    ∂g∂y = _calculate_∂g∂y(implicit_problem, x, y, buffers)
    ∂Y∂X(lu!(∂g∂y))
end

"""
`$(SIGNATURES) → nothing`

Calculate the pushforward `dy = ∂y∂x ⋅ dx` into `dy`.

A fallback is provided using Enzyme, but an `implicit_problem` can define its own method.
"""
function calculate_pushforward!(dy, implicit_problem, x, y, ∂y∂x::∂Y∂X, dx)
    @argcheck get_solution_concept(implicit_problem) ≡ ZeroResiduals()
    buffers = get_buffers(implicit_problem)
    _inplace_∂g∂x_v!(dy, dx, implicit_problem, x, y, buffers.r)
    ldiv!(∂y∂x.∂g∂y_factor, dy)
    dy .*= -1
    nothing
end

"""
`$(SIGNATURES)` → `nothing`

Accumulate the pullback `dy ⋅ ∂y∂x` into `dx`.

A default is implemented using Enzyme, but an `implicit_problem` can define its own method.
"""
function accumulate_pullback!(dx, implicit_problem, x, y, ∂y∂x::∂Y∂X, dy)
    @argcheck get_solution_concept(implicit_problem) ≡ ZeroResiduals()
    (; ∂g∂y_factor) = ∂y∂x
    # math:
    #     dy ⋅ ∂y/∂x = - (dy' / ∂g/∂y) ⋅ ∂g/∂x
    buffers = get_buffers(implicit_problem)
    buffers.y .= dy                # buffer_y1 == dy
    rdiv!(buffers.y', ∂g∂y_factor) # a == dy' / ∂g∂y
    _inplace_v_∂g∂x!(buffers.x, buffers.y, implicit_problem, x, y,
                     buffers.r)  # b = (dy' / ∂g/∂y) ⋅ ∂g/∂x
    dx .-= buffers.x
    nothing
end
