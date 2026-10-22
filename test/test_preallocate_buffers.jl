@testset "preallocate buffers" begin
    @test E.API_sanity_checks(E.preallocate_buffers(LinearProblem(; n_x = 4, n_y = 3))).all_ok
end
