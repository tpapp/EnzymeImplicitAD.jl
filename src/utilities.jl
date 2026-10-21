#####
##### utilities
#####

####
#### Primitives for calculating Jacobians and Jacobian-vector products using Enzyme.
#### Buffers need to be provided explicitly, higher-level API manages them.
####

"Default chunks for calculating the Jacobian when the dimensions are larger than the threshold."
const AD_CHUNKS = 4

function _calculate_∂g∂y_onecol!(J, implicit_problem, x::AbstractVector, y::AbstractVector{T},
                                 buffers) where T
    # aliases for buffers
    (; r) = buffers
    dr = buffers.dr[1]
    dy = buffers.dy[1]
    # calculate Jacobian column by column
    make_zero!(dy)
    for i in axes(y, 1)
        make_zero!(dr)          # FIXME do I need this?
        dy[i] = one(T)
        autodiff(Forward, implicit_residuals!, Duplicated(r, dr),
                 Const(implicit_problem), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
    J
end

function _calculate_∂g∂y_chunked!(J, implicit_problem, x::AbstractVector, y::AbstractVector{T},
                                  chunks::Val{C}, buffers) where {T,C}
    (; n_y) = get_dimensions(implicit_problem)
    @assert C ≤ n_y
    (; r, dy, dr) = buffers
    @assert length(dy) ≤ C
    @assert length(dr) ≤ C
    make_zero!(dy)
    function _set_dy(i, v)
        for j in 1:C
            dy[j][i-1+j] = v
        end
        nothing
    end
    function _calc_J(i)
        make_zero!(dr)
        _set_dy(i, one(T))
        autodiff(Forward, implicit_residuals!, BatchDuplicated(r, dr),
                 Const(implicit_problem), Const(x), BatchDuplicated(y, dy))
        for j in 1:C
            J[:, i-1+j] .= dr[j]
        end
        _set_dy(i, zero(T))
    end
    for i in 1:C:n_y
        i = min(i, n_y - C + 1)   # overlap
        _calc_J(i)
    end
    J
end

"""
`$(SIGNATURES)` → `J`

Calculate the Jacobian ``J = ∂g/∂y``, at `x` and `y`, which is assumed to be a valid
solution (not checked).

See [`get_buffers`](@ref) about `buffers`.
"""
function _calculate_∂g∂y(implicit_problem, x, y, buffers)
    @assert !Base.has_offset_axes(x, y)
    (; n_y, n_r) = get_dimensions(implicit_problem)
    J = similar(y, n_r, n_y)
    if n_y ≥ 8
        ∂g∂y = _calculate_∂g∂y_chunked!(J, implicit_problem, x, y, Val(AD_CHUNKS), buffers)
    else
        ∂g∂y = _calculate_∂g∂y_onecol!(J, implicit_problem, x, y, buffers)
    end
    J
end

"""
`$(SIGNATURES)` → `nothing`

Calculate `∂g/∂x ⋅ v` and put the result in the first argument, using forward mode in Enzyme.

See [`get_buffers`](@ref) about `buffer_r`.

### Expected dimensions (not checked):

- `length(v) == n_x`
- `length(Jv) == n_r`
"""
function _inplace_∂g∂x_v!(Jv, v, implicit_problem, x, y, buffer_r)
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, implicit_residuals!, Duplicated(buffer_r, Jv), Const(implicit_problem),
             Duplicated(x, v), Const(y))
    nothing
end

"""
`$(SIGNATURES)` → `nothing`

Calculate `v ⋅ ∂g/∂x` and put the result in the first argument, using reverse mode in Enzyme.

**Caution:** may modify `v`.

See [`get_buffers`](@ref) about `buffer_r`.

### Expected dimensions (not checked):

- `length(v) == n_r`
- `length(Jv) == n_x`
"""
function _inplace_v_∂g∂x!(vJ, v, implicit_problem, x, y, buffer_r)
    make_zero!(vJ)              # FIXME: do I need this?
    autodiff(Reverse, implicit_residuals!, Const,
             Duplicated(buffer_r, v), Const(implicit_problem), Duplicated(x, vJ), Const(y))
    nothing
end

####
#### for statistics
####

"""
An online mean implementation. Use [`online_mean`](@ref) as the entry point.
"""
mutable struct OnlineMean{T}
    count::Int
    sum::T
end

independent_copy(om::OnlineMean) = OnlineMean(om.count, om.sum)

function Base.show(io::IO, om::OnlineMean)
    (; sum, count) = om
    print(io, "« $(sum) / $(count) ≈ ",
          count == 0 ? "undefined" : "$(round(sum / count; digits = 4))", " »")
end

"""
`$(SIGNATURES)` → `accumulator`

Return an accumulator that supportes [`update!`](@ref) and [`get_mean`](@ref).
The sum is accumulated in a value of type `T`.
"""
online_mean(::Type{T}) where T = OnlineMean(0, zero(T))

function update!(om::OnlineMean{T}, x) where T
    om.count += 1
    om.sum += T(x)
    nothing
end

get_mean(om::OnlineMean) = om.sum / om.count

####
#### namedtuple utilities
####

"""
`$(SIGNATURES)` → `nothing`

Merge two `NamedTuple`s, throw an error if they have names in common.
"""
function merge_disjoint(a::NamedTuple, b::NamedTuple)
    ab = merge(a, b)
    if length(a) + length(b) ≠ length(ab)
        throw(ArgumentError("found common keys " *
            join(intersect(keys(a), keys(b)), ", ")))
    end
    ab
end
