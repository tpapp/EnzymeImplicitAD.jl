@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n_x, n_y = 3, 4
    (; A, B) = P = LinearProblem(; n_y, n_x)
    x = randn(n_x)
    y = randn(n_y)
    dy = similar(y)
    b1 = similar(y)
    b2 = similar(y)
    b3 = similar(y)

    J = E._calculate_∂g∂y(P, x, y, b1, b2, b3)
    @test J ≈ B

    Jv = similar(y)
    v = randn(n_x)
    E._inplace_∂g∂x_v!(Jv, v, P, x, y, b1)
    @test Jv ≈ A * v

    vJ = similar(x)
    v = randn(n_y)
    E._inplace_v_∂g∂x!(vJ, copy(v), P, x, y, b1)
    @test vJ ≈ A' * v
end
