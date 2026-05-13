"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

public implicit_solve!, implicit_residuals!

using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using ConcreteStructs: @concrete
import LinearAlgebra: mul!

include("api.jl")
include("enzyme_ad.jl")
include("solvers.jl")

end # module
