#####
##### utilities
#####

####
#### Primitives for calculating Jacobians and Jacobian-vector products using Enzyme.
#### Buffers need to be provided explicitly, higher-level API manages them.
####

const BUFFER_DOCS = "`buffer_…` variables are for temporary storage, may be overwritten, their initial contents can be arbitrary. It is assumed that they are not shared between tasks."

"""
$(SIGNATURES)

Calculate the Jacobian `J = ∂g/∂y`, at `x` and `y`, which is assumed to be a valid
solution (not checked).

$(BUFFER_DOCS)
"""
function _calculate_∂g∂y(implicit_problem, x::AbstractVector, y::AbstractVector{T},
                         buffer_y1, buffer_y2, buffer_y3) where T
    # aliases for buffers
    dy = buffer_y1
    r = buffer_y2
    dr = buffer_y3
    # calculate Jacobian column by column
    make_zero!(dy)
    J = similar(y, T, axes(y, 1), axes(x, 1))
    for i in axes(x, 1)
        make_zero!(dr)          # FIXME do I need this?
        dy[i] = one(T)
        autodiff(Forward, implicit_residuals!, Duplicated(r, dr),
                 Const(implicit_problem), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
    J
end

"""
$(SIGNATURES)

Calculate `∂g/∂x ⋅ v` and put the result in the first argument, using forward mode in Enzyme.

$(BUFFER_DOCS)
"""
function _inplace_∂g∂x_v!(Jv, v, implicit_problem, x, y, buffer_y1)
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, implicit_residuals!, Duplicated(buffer_y1, Jv), Const(implicit_problem),
             Duplicated(x, v), Const(y))
    nothing
end

"""
$(SIGNATURES)

Calculate `v ⋅ ∂g/∂x` and put the result in the first argument, using reverse mode in Enzyme.

**Caution:** may modify `v`.

$(BUFFER_DOCS)
"""
function _inplace_v_∂g∂x!(vJ, v, implicit_problem, x, y, buffer_y1)
    make_zero!(vJ)              # FIXME: do I need this?
    autodiff(Reverse, implicit_residuals!, Duplicated(buffer_y1, v), Const(implicit_problem),
             Duplicated(x, vJ), Const(y))
    nothing
end
