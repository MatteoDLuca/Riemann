# RCoR scatterplot on synthetic SPD matrices.
# Julia 1.10+; install PosDefManifold, Diagonalizations and Plots first.
#
# Shows raw matrices, registered matrices, the raw arithmetic mean and RCoR reference.
# Embedding: classical MDS of affine-invariant (Fisher) SPD distances.

using LinearAlgebra, DelimitedFiles, Statistics
using PosDefManifold, Diagonalizations, Plots

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "julia", "RR_nuclear_optimized_final.jl"))

function fisher_distance(A::Matrix{Float64}, B::Matrix{Float64})
    decomposition = eigen(Symmetric(A))
    minimum(decomposition.values) > 0 || error("Not SPD")
    invhalf = decomposition.vectors * Diagonal(1.0 ./ sqrt.(decomposition.values)) * decomposition.vectors'
    λ = eigvals(Symmetric(invhalf * B * invhalf))
    minimum(λ) > 0 || error("Not SPD")
    return norm(log.(λ))
end

function fisher_mds(C::Vector{Matrix{Float64}})
    k = length(C)
    d2 = zeros(Float64, k, k)
    for i in 1:k, j in (i+1):k
        d2[i, j] = d2[j, i] = fisher_distance(C[i], C[j])^2
    end
    H = Matrix{Float64}(I, k, k) .- 1.0 / k
    gram = Symmetric(-0.5 .* (H * d2 * H))
    E = eigen(gram)
    indices = sortperm(E.values, rev=true)[1:2]
    return E.vectors[:, indices] * Diagonal(sqrt.(max.(E.values[indices], 0.0)))
end

function run_demo(case::String="demo_variable_spectra_n12_d8";
                  output::String=joinpath(ROOT, "examples", "output", "rcor_scatter_julia.png"))
    folder = joinpath(ROOT, "examples", "data", case)
    if !isdir(folder)
        folder = joinpath(ROOT, "fixtures", case)
    end
    filenames = sort(filter(f -> startswith(f, "C_") && endswith(f, ".csv"), readdir(folder)))
    isempty(filenames) && error("No synthetic CSV fixtures in $folder")
    raw = [Matrix{Float64}(readdlm(joinpath(folder, f), ',')) for f in filenames]

    # Apply the RCoR implementation.
    M, Γ, 𝐆 = rr(ℍVector([ℍ(C) for C in raw]))
    registered = [Matrix(G) for G in 𝐆]
    arithmetic_mean = sum(raw) / length(raw)
    coords = fisher_mds([raw; registered; [arithmetic_mean, Matrix(M)]])
    n = length(raw)

    p = scatter(coords[1:n, 1], coords[1:n, 2];
        label="Raw SPD matrices", color=:steelblue, marker=:circle,
        markersize=6, xlabel="Fisher-distance MDS component 1",
        ylabel="Fisher-distance MDS component 2",
        title="Riemannian Co-Registration | synthetic SPD example",
        legend=:best, size=(950, 620), dpi=170)
    for i in 1:n
        plot!(p, coords[[i, n+i], 1], coords[[i, n+i], 2];
              color=:gray, alpha=0.25, linewidth=0.8, label=false)
    end
    scatter!(p, coords[n+1:2n, 1], coords[n+1:2n, 2];
        label="Registered SPD matrices", color=:seagreen, marker=:rect, markersize=6)
    scatter!(p, [coords[2n+1, 1]], [coords[2n+1, 2]];
        label="Raw arithmetic mean", color=:orange, marker=:diamond, markersize=9)
    scatter!(p, [coords[2n+2, 1]], [coords[2n+2, 2]];
        label="RCoR reference M", color=:gold, marker=:star5, markersize=13)

    mkpath(dirname(output))
    savefig(p, output)
    println("Saved figure: ", output)
    println("Input: ", n, " synthetic ", size(raw[1], 1), "x", size(raw[1], 2), " SPD matrices")
    println("Gamma orthogonality error: ", norm(Γ' * Γ - I))
    return output
end

run_demo(isempty(ARGS) ? "demo_variable_spectra_n12_d8" : ARGS[1])
