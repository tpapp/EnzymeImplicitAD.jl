"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

public implicit_solve!, implicit_residuals!

using DocStringExtensions: FUNCTIONNAME, SIGNATURES

include("api.jl")
include("enzyme_ad.jl")

end # module
