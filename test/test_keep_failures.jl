@testset "keep_failures" begin
    @test E.API_sanity_checks(LinearProblem(; n_x = 4, n_y = 3)).all_ok

    P = E.keep_failures(MayFail())
    N = 100
    failures = Vector{Vector{Float64}}()
    y = [NaN]
    for _ in 1:N
        x = [randn()]
        success = E.implicit_solve!(y, P, x)
        if !success
            push!(failures, copy(x))
        end
    end
    stats = E.get_statistics(P)
    @test stats.failure_rate ≈ length(failures) / N
    @test sort(failures) == sort(stats.failures)
end
