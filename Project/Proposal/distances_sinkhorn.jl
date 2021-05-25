using UMAP: umap
using MLDatasets
using VegaLite
using DataFrames
using OptimalTransport
# using Tulip
using Distances
using LinearAlgebra
using CSV
using JLD2
using CUDA

x = MNIST.traintensor(Float64)
mnist_x = reshape(x, 28*28, :);
mnist_y = MNIST.trainlabels(1:size(mnist_x, 2));

normalize_mnist(x) = mnist_x[:,x] / sum(mnist_x[:,x])
norm_mnist_x = normalize_mnist.(1:length(mnist_x[1,:]));

N = 5000
n = 28
nonzero = []
for i in norm_mnist_x
    push!(nonzero,findall(!iszero,i))
end
distm = collect(Iterators.product(1:n,1:n));
μ = reshape(distm,n*n);
ν = reshape(distm,n*n);
@time jldopen("mnist_distances_sinkhorn10.jld2", "w+") do file
    Threads.@threads for i in 1:N
        D = zeros(N)
        for j in i+1:N
            u = norm_mnist_x[i]
            v = norm_mnist_x[j]
            C = float.(pairwise(SqEuclidean(), μ[nonzero[i]], ν[nonzero[j]]))
            u = u[nonzero[i]]
            v = v[nonzero[j]]
            D[j] = sinkhorn2(u,v,C,10)
        end
        file["row"*string(i)] = D
    end
end

D = UpperTriangular(zeros(N,N))
jldopen("mnist_distances_sinkhorn10.jld2", "r") do file
    for i in 1:N
        D[i,:] = file["row"*string(i)]
    end
end

Dfull = Symmetric(D);

res_jl = umap(Dfull,
    2; metric=:precomputed, n_neighbors=10, min_dist=0.001, n_epochs=200)
