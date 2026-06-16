#####
##### wrapper to cache y and ∂y∂x
#####

using ThreadSafeDicts: ThreadSafeDict

"""
$(SIGNATURES)

Helper function for consistent value types in dictionaries.
"""
@inline function _cache_value_type(Y,∂Y∂X)
    @NamedTuple{timestamp::Int64,y::Y,∂y∂x::Union{Nothing,∂Y∂X}}
end

# NOTE: parametrization assumes `x` and `y` values have the same type
@concrete struct CacheImplicitProblem{Y,∂Y∂X,D<:AbstractDict{Y,_cache_value_type(Y,∂Y∂X)}}
    inner_problem
    min_size::Int
    max_size::Int
    dict::D
    y_hits
    ∂y∂x_hits
    function CacheImplicitProblem(inner_problem, min_size::Int, max_size::Int)
        @argcheck 0 < min_size < max_size
        T = get_preferred_eltype(inner_problem)
        Y = Vector{T}
        ∂Y∂X = get_∂y∂x_type(inner_problem)
        dict = ThreadSafeDict{Y,_cache_value_type(Y,∂Y∂X)}()
        y_hits = online_mean(UInt64)
        ∂y∂x_hits = online_mean(UInt64)
        # {Y,∂Y∂X,typeof(dict),            typeof(inner_problem)}
        new{Y,∂Y∂X,typeof(dict),typeof(inner_problem),
            typeof(y_hits),typeof(∂y∂x_hits)}(inner_problem, min_size, max_size, dict,
                                              y_hits, ∂y∂x_hits)
    end
end

function Base.show(io::IO, problem::CacheImplicitProblem)
    (; min_size, max_size, inner_problem) = problem
    print(io, "caching [$(min_size),$(max_size)] evaluations of $(inner_problem)")
end

for f in [:get_dimensions, :get_preferred_eltype, :task_local_buffers, :get_∂y∂x_type]
    @eval ($f)(implicit_problem::CacheImplicitProblem) = ($f)(implicit_problem.inner_problem)
end

function get_statistics(problem::CacheImplicitProblem)
    (; inner_problem, y_hits, ∂y∂x_hits) = problem
    merge_disjoint(get_statistics(inner_problem),
                   (; average_y_hit = get_mean(y_hits),
                    average_∂y∂x_hit = get_mean(∂y∂x_hits)))
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
    CacheImplicitProblem(inner_problem, min_size, max_size)
end

function _cull!(dict, min_size)
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
    (; inner_problem, min_size, max_size, dict, y_hits) = implicit_problem
    timestamp = time_ns()
    if haskey(dict, x)
        (; y, ∂y∂x) = dict[x]
        dict[x] = (; timestamp, y, ∂y∂x)
        y_hit = true
    else
        (; n_y) = get_dimensions(inner_problem)
        y = Vector{get_preferred_eltype(inner_problem)}(undef, n_y)
        implicit_solve!(y, inner_problem, x)
        dict[_ensure_typed_copy(Y, x)] = (; timestamp, y, ∂y∂x = nothing)
        length(dict) > max_size && _cull!(dict, min_size)
        y_hit = false
    end
    update!(y_hits, y_hit)
    copy!(y2, y)
    nothing
end

function calculate_∂y∂x(implicit_problem::CacheImplicitProblem{Y}, x, _y) where Y
    # NOTE: the y argument is ignored, it is obtained from the cache
    (; inner_problem, min_size, max_size, dict, y_hits, ∂y∂x_hits) = implicit_problem
    timestamp = time_ns()
    if haskey(dict, x)
        (; y, ∂y∂x) = dict[x]
        if ∂y∂x ≡ nothing
            ∂y∂x = calculate_∂y∂x(inner_problem, x, y)
            update!(y_hits, true)
            ∂y∂x_hit = false
        else
            ∂y∂x_hit = true
        end
        dict[x] = (; timestamp, y, ∂y∂x)
        update!(∂y∂x_hits, ∂y∂x_hit)
    else
        (; n_y) = get_dimensions(inner_problem)
        T = get_preferred_eltype(inner_problem)
        y = Vector{T}(undef, n_y)
        implicit_solve!(y, inner_problem, x)
        ∂y∂x = calculate_∂y∂x(inner_problem, x, y)
        dict[_ensure_typed_copy(Y, x)] = (; timestamp, y, ∂y∂x)
        length(dict) > max_size && _cull!(dict, min_size)
        update!(y_hits, false)
        update!(∂y∂x_hits, false)
    end
    ∂y∂x
end
