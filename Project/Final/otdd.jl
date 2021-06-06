using OptimalTransport
using VegaLite
using Distances
using Distributions
using Images
using DataFrames
using LinearAlgebra
using SparseArrays
using Tulip
using UMAP


"""
    _fit_MvNormals(X,y)
`X` is the feature dataset, where rows are the samples and the columns
are feateus. Vector `y` is the label vector.
"""
function _fit_MvNormals(X,y)
    labels = sort(unique(y))
    α = []
    for i in labels
        Σ = cov(X[y .== i,:]);
        Σ += max(0,-2*eigmin(Σ))*I
        m = vec(mean(X[y .== i, :],dims=1))
        push!(α, MvNormal(m,Σ))
    end
    return α
end

"""
_getW(X1, y1, X2, y2)
Calculate the ``W^2_2(\\alpha_1(i), \\alpha_2(j))`` distance for all combinations
of pair os labels, where ``\\alpha_1(i) := P(X1 | y1 = i)``
and ``\\alpha_2(j) := P(X2 | y2 = j)``.
"""
function _getW(X1, y1, X2, y2)
    α1 = _fit_MvNormals(X1, y1)
    α2 = _fit_MvNormals(X2, y2)
    
    Wα = zeros(length(α1),length(α2))
    for (i,j) in Iterators.product(1:length(α1),1:length(α2))
        Wα[i,j] = ot_cost(SqEuclidean(), α1[i],α2[j])
    end
    return Wα
end

"""
    otdd(dx, D1, D2, ε = 0.1)
Calculates the Optimal Transport Dataset Distance between
datasets D1 and D2, using metric dx between features
distance. Note that `dx` must be a `PreMetric` from `Distances.jl`.
"""
function otdd(X1, y1, X2, y2; ε = 1, dx=SqEuclidean(), W = nothing)
    if W === nothing
        α1 = _fit_MvNormals(X1, y1)
        α2 = _fit_MvNormals(X2, y2)

        # store the 2-Wasserstein distance
        # between P(X1 | y1 = i) and P(X2 | y2 = j)
        W = zeros(length(α1),length(α2))
        for (i,j) in Iterators.product(1:length(α1),1:length(α2))
            W[i,j] = ot_cost(SqEuclidean(), α1[i],α2[j])
        end
    end
    
    C = pairwise(dx, X1, X2, dims=1);
    for (i,j) in Iterators.product(1:N,1:N)
        C[i,j] = sqrt(C[i,j] + W[y1[i]+1,y2[j]+1])
    end
    
    n1 = length(y1)
    n2 = length(y2)
    γ = sinkhorn(ones(n1)./n1, ones(n2)./n2, C, ε);
    
    return C, γ, dot(C,γ)
end