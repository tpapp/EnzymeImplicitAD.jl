"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, FIELDS, TYPEDEF
import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, BatchDuplicated, FwdConfig, RevConfigWidth, overwritten,
    AugmentedReturn, needs_primal
using Enzyme: Forward, Reverse, autodiff, make_zero!
using LinearAlgebra: ldiv!, lu!, rdiv!

include("utilities.jl")
include("api.jl")
include("sanity_checks.jl")
include("benchmarks.jl")
include("enzyme_ad.jl")
include("cache.jl")
include("keep_failures.jl")

end # module
