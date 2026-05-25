#####
##### wrapper to cache y and ∂y∂x
#####

"""
$(SIGNATURES)

Helper function for consistent value types in dictionaries.
"""
@inline function _cache_value_type(Y,∂Y∂X)
    @NamedTuple{timestamp::Int64,y::Y,∂y∂x::Union{Nothing,∂Y∂X}}
end

# NOTE: parametrization assumes `x` and `y` values have the same type
struct CacheImplicitProblem{Y,∂Y∂X,P}
    inner_problem::P
    min_size::Int
    max_size::Int
    dict::Dict{Y,_cache_value_type(Y,∂Y∂X)}
end

for f in [:get_dimensions, :get_preferred_eltype, :task_local_buffers, :get_∂y∂x_type]
    @eval ($f)(implicit_problem::CacheImplicitProblem) = ($f)(implicit_problem.inner_problem)
end

function implicit_residuals!(r, implicit_problem::CacheImplicitProblem, x, y)
    implicit_residuals!(r, implicit_problem.inner_problem, x, y)
end

"""
$(SIGNATURES)

Wrap an implicit problem so that `y` and `∂y∂x` are cached.

Specficially, at least `min_size` and at most `max_size` most recently used values are kept.
"""
function cache_implicit_problem(inner_problem::P;
                                min_size::Int = 10, max_size = 2 * min_size) where P
    @argcheck 0 < min_size < max_size
    T = get_preferred_eltype(inner_problem)
    Y = Vector{T}
    ∂Y∂X = get_∂y∂x_type(inner_problem)
    CacheImplicitProblem{Y,∂Y∂X,P}(inner_problem, min_size, max_size,
                                   Dict{Y,_cache_value_type(Y,∂Y∂X)}())
end

function _cull!(implicit_problem::CacheImplicitProblem)
    (; min_size, dict) = implicit_problem
    timestamps = [x.timestamp for x in values(dict)]
    sort!(timestamps; rev = true)
    cutoff = timestamps[min_size]
    for (k, v) in pairs(dict)
        if v.timestamp < cutoff
            delete!(dict, k)
        end
    end
    nothing
end

_ensure_typed_copy(::Type{X}, x::X) where X = copy(x)

_ensure_typed_copy(::Type{_X}, x::X) where {_X,X} = _X(x)

function implicit_solve!(y2, implicit_problem::CacheImplicitProblem{Y}, x) where Y
    (; inner_problem, max_size, dict) = implicit_problem
    timestamp = time_ns()
    if haskey(dict, x)
        (; y, ∂y∂x) = dict[x]
        dict[x] = (; timestamp, y, ∂y∂x)
    else
        (; n_y) = get_dimensions(inner_problem)
        y = Vector{get_preferred_eltype(inner_problem)}(undef, n_y)
        implicit_solve!(y, inner_problem, x)
        dict[_ensure_typed_copy(Y, x)] = (; timestamp, y, ∂y∂x = nothing)
        length(dict) > max_size && _cull!(implicit_problem)
    end
    copy!(y2, y)
    nothing
end

function calculate_∂y∂x(implicit_problem::CacheImplicitProblem{Y}, x, _y) where Y
    # NOTE: the y argument is ignored, it is obtained from the cache
    (; inner_problem, max_size, dict) = implicit_problem
    timestamp = time_ns()
    if haskey(dict, x)
        (; y, ∂y∂x) = dict[x]
        if ∂y∂x ≡ nothing
            ∂y∂x = calculate_∂y∂x(inner_problem, x, y)
        end
        dict[x] = (; timestamp, y, ∂y∂x)
    else
        (; n_y) = get_dimensions(inner_problem)
        T = get_preferred_eltype(inner_problem)
        y = Vector{T}(undef, n_y)
        impicit_solve!(y, inner_problem, x)
        ∂y∂x = calculate_∂y∂x(inner_problem, x, y)
        dict[_ensure_typed_copy(Y, x)] = (; timestamp, y, ∂y∂x)
        length(dict) > max_size && _cull!(cache)
    end
    ∂y∂x
end
