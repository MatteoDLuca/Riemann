using LinearAlgebra, PosDefManifold, Diagonalizations

# Use all available BLAS threads for BLAS/LAPACK-based operations.
BLAS.set_num_threads(Sys.CPU_THREADS)


# ---------------------------------------------------------------------
# Nuclear norm
# ---------------------------------------------------------------------
#
# True nuclear norm:
#
#     ||A||_* = sum_k sigma_k(A)
#
# Since the eigenvalues of A*A' are the squared singular values of A,
# compute:
#
#     ||A||_* = sum_k sqrt(lambda_k(A*A'))
#
# This avoids the full SVD used in sum(svd(A).S).
function nuclearnorm(A::AbstractMatrix{T}) where T<:Real
    λ = eigvals(Symmetric(A * A'))
    s = zero(eltype(λ))

    @simd for k in eachindex(λ)
        @inbounds s += sqrt(max(λ[k], zero(eltype(λ))))
    end

    return s
end


# ---------------------------------------------------------------------
# Signed sum of eigenvector matrices
# ---------------------------------------------------------------------
#
# Given eigenvector matrices U_i and sign vectors s_i, compute:
#
#     A(S) = sum_i U_i S_i
#
# where each S_i is represented by the vector s_i containing its diagonal.
function _signed_sum(𝐔::Vector{Matrix{Float64}},
                     𝐬::Vector{Vector{Float64}})
    n = length(𝐔)
    d = size(𝐔[1], 1)
    A = zeros(Float64, d, d)

    @inbounds for i = 1:n
        U = 𝐔[i]
        s = 𝐬[i]

        for j = 1:d
            sij = s[j]

            @simd for r = 1:d
                @inbounds A[r, j] += sij * U[r, j]
            end
        end
    end

    return A
end


# ---------------------------------------------------------------------
# Global column-wise sign synchronization
# ---------------------------------------------------------------------
#
# Initialization step.
#
# For each column j, solve exactly:
#
#     max_{s_2j,...,s_nj} || U_1[:,j] + sum_{i=2}^n s_ij U_i[:,j] ||^2
#
# with s_1j fixed to +1 to remove the global sign ambiguity.
#
# This is exactly the maximization of the Frobenius proxy
#
#     sumOfSqr(A(S)) = ||A(S)||_F^2
#
# because ||A(S)||_F^2 separates column by column.
function _columnsync_init(𝐔::Vector{Matrix{Float64}})
    n = length(𝐔)
    d = size(𝐔[1], 1)

    𝐬 = [ones(Float64, d) for i = 1:n]
    n == 1 && return 𝐬

    v = zeros(Float64, d)
    npatterns = Int(2)^(n - 1)

    for j = 1:d

        bestval = -Inf
        bestmask = 0

        for mask = 0:(npatterns - 1)

            fill!(v, 0.0)

            @inbounds for i = 1:n

                si = if i == 1
                    1.0
                else
                    ((mask >> (i-2)) & 1) == 1 ? -1.0 : 1.0
                end

                U = 𝐔[i]

                @simd for r = 1:d
                    @inbounds v[r] += si * U[r, j]
                end
            end

            # PosDefManifold.jl:
            # sumOfSqr(v) = sum_r v[r]^2
            val = sumOfSqr(v)

            if val > bestval
                bestval = val
                bestmask = mask
            end
        end

        @inbounds 𝐬[1][j] = 1.0

        @simd for i = 2:n
            @inbounds 𝐬[i][j] = ((bestmask >> (i-2)) & 1) == 1 ? -1.0 : 1.0
        end
    end

    return 𝐬
end


# ---------------------------------------------------------------------
# In-place flip of one sign
# ---------------------------------------------------------------------
#
# Flipping s_ij changes only column j of the signed sum A:
#
#     A_new[:,j] = A_old[:,j] - 2*s_ij*U_i[:,j]
function _flip_column!(A::Matrix{Float64},
                       U::Matrix{Float64},
                       sij::Float64,
                       j::Int)
    d = size(A, 1)
    α = -2.0 * sij

    @simd for r = 1:d
        @inbounds A[r, j] += α * U[r, j]
    end

    return A
end


# ---------------------------------------------------------------------
# Greedy coordinate ascent on the true nuclear norm
# ---------------------------------------------------------------------
#
# Starting from the column-synchronized signs, try one sign flip at a time.
# Accept a flip only if it increases the true nuclear norm of A(S).
#
# The first matrix is kept fixed as gauge, so the loop starts from i = 2.
function _nuclear_greedy!(𝐔::Vector{Matrix{Float64}},
                          𝐬::Vector{Vector{Float64}},
                          A::Matrix{Float64};
                          tol::Float64 = 1e-12,
                          maxiter::Int = 100)
    n = length(𝐔)
    d = size(𝐔[1], 1)

    f = nuclearnorm(A)

    for iter = 1:maxiter
        improved = false

        # The order is important because every accepted flip updates A.
        # Therefore use @inbounds, but not @simd on these loops.
        @inbounds for i = 2:n
            U = 𝐔[i]
            s = 𝐬[i]

            for j = 1:d
                sij = s[j]

                # Try the flip.
                _flip_column!(A, U, sij, j)
                fcand = nuclearnorm(A)

                if fcand > f + tol
                    # Accept the flip.
                    s[j] = -sij
                    f = fcand
                    improved = true
                else
                    # Reject the flip and restore A.
                    _flip_column!(A, U, -sij, j)
                end
            end
        end

        improved || break
    end

    return 𝐬, A, f
end


# ---------------------------------------------------------------------
# Riemannian Registration / Co-Registration with nuclear-norm sign fixing
# ---------------------------------------------------------------------
#
# This keeps the same structure as RR.jl:
#
#   1. eigenvalue decomposition of each SPD matrix;
#   2. construction of a common basis Gamma from the SVD of the sum
#      of eigenvector matrices;
#   3. geometric mean of eigenvalues;
#   4. reconstruction of the reference SPD matrix and registered matrices.
#
# The difference is that the sum of eigenvector matrices is now a signed
# sum. The signs are obtained by:
#
#   a. exact global column-wise synchronization using the sumOfSqr proxy;
#   b. greedy refinement using the true nuclear norm.
function rr(𝐂::ℍVector; tol::Float64 = 1e-12, maxiter::Int = 100)
    n = length(𝐂)

    𝛌 = Vector{Vector{Float64}}(undef, n)
    𝐔 = Vector{Matrix{Float64}}(undef, n)

    # Same eigenvalue decomposition as in the original RR.jl.
    for (i, C) ∈ enumerate(𝐂)
        𝛌[i], 𝐔[i] = eig(C)
    end

    d = length(𝛌[1])

    # Step 1: sign initialization.
    # This already optimizes the Frobenius/sumOfSqr proxy column by column.
    𝐬 = _columnsync_init(𝐔)

    # Step 2: construct the signed sum.
    A = _signed_sum(𝐔, 𝐬)

    # Step 3: greedy refinement with the true nuclear norm.
    _nuclear_greedy!(𝐔, 𝐬, A; tol = tol, maxiter = maxiter)

    # Same construction as in the original RR.jl:
    # SVD of the final signed sum.
    sv = svd(A)

    # Polar factor: Gamma = sv.U * sv.Vt
    Γ = BLAS.gemm('N', 'N', sv.U, sv.Vt)

    # Geometric mean of eigenvalues.
    𝓁 = [exp(sum(log((𝛌[i][j])) for i = 1:n) / n) for j = 1:d]

    # Same output structure as the original rr function.
    return ℍ(Γ * (𝓁 .* Γ')),
           Γ,
           ℍVector([ℍ(Γ * (λ .* Γ')) for λ ∈ 𝛌])
end
