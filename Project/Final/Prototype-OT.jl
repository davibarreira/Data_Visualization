### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 8529c382-f72f-44f7-8ccd-ce68ab03776e
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="PlutoUI"),
        Pkg.PackageSpec(name="HypertextLiteral"),
        Pkg.PackageSpec(name="JSON"),
		Pkg.PackageSpec(name="UMAP"),
		Pkg.PackageSpec(name="MLDatasets"),
		Pkg.PackageSpec(name="VegaLite"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="OptimalTransport"),
		Pkg.PackageSpec(name="Distances"),
		Pkg.PackageSpec(name="LinearAlgebra"),
		Pkg.PackageSpec(name="JSONTables"),
		Pkg.PackageSpec(name="Images")
    ])
    using UMAP, MLDatasets, VegaLite, DataFrames, OptimalTransport, Distances, LinearAlgebra, PlutoUI, HypertextLiteral, JSON, JSONTables, Images
end

# ╔═╡ 2ffddf10-bd51-11eb-12cb-f1add38b47fb
md"""
# Optimal Transport for Dataset Distances to Aid Transfer Learning
"""

# ╔═╡ b3a49e8b-b54c-4247-8370-c2a917e57056
md"""
### Installing and Importing Packages and Data
"""

# ╔═╡ 4ff7b77e-27e7-4034-8939-07cc319712ad
mnist_x = reshape(MNIST.traintensor(Float64),28*28,:);

# ╔═╡ 73b2d14c-1f89-425c-92a0-57591f21e8fd
mnist_y = MNIST.trainlabels(1:size(mnist_x, 2));

# ╔═╡ f411e2a4-66b2-41e0-90c0-a948cd6789f0
fmnist_x = reshape(FashionMNIST.traintensor(Float64),28*28,:);

# ╔═╡ f2c0afcc-016a-4802-b75b-80d258b45af6
fmnist_y = FashionMNIST.trainlabels(1:size(fmnist_x, 2));

# ╔═╡ b3fd7749-18ef-4033-9e2d-431ee284c11b
md"""
### Data Wrangling
"""

# ╔═╡ 4a741e16-7e80-43cb-bfe3-63ae442d61f1
md"""
Sample size for testing 
"""

# ╔═╡ dc2c4b08-9c1c-4555-bf7b-60be8e6bd0d9
N = 100

# ╔═╡ 75b36234-7d5b-4527-9274-2239046b556a
md"""
Applying UMAP to reduce dimensionality of the datasets
"""

# ╔═╡ 068369ca-a6db-4f01-b192-1256332202f0
res_jl = umap(hcat(mnist_x[:,1:N],fmnist_x[:,1:N]); n_neighbors=10, min_dist=0.001, n_epochs=200);

# ╔═╡ b89edb81-2e62-4b2a-8ce3-3e4c25a31b55
md"""
Generating dataset for plots
"""

# ╔═╡ 23c1d16f-d23c-4a99-b4c9-db3ad782e8fc
img_url = vcat(
    ["./images/mnist_"*string(i)*".png" for i in 1:N],
    ["./images/fmnist_"*string(i)*".png" for i in 1:N]);

# ╔═╡ df0f24fd-f847-40fb-b3dc-12350face55f
df = DataFrame(
    x1     = res_jl'[:,1],
    x2     = res_jl'[:,2],
    img    = img_url,
    label  = vcat(mnist_y[1:N],fmnist_y[1:N]),
    dataset= vcat(["mnist" for i in 1:N],["fmnist" for i in 1:N]));

# ╔═╡ 29ac65a4-a1c1-47a8-a691-be90f988709f
md"""
Dataframe to Json to pass to JavaScript
"""

# ╔═╡ 3994768a-526e-4116-8dee-f398c7a36ffd
dfjson = arraytable(df);

# ╔═╡ 21b3b741-1ea1-49a4-a6ae-b22666f53e19
md"""
Dataset for heatmap
"""

# ╔═╡ 92af2802-acb2-4b67-9449-1fe793174df7
source = df[1:N,:];

# ╔═╡ 32a4e97d-ecf0-4ff0-9d2d-0d6f47b855f2
source[!,:fmnist_label] = df[source[!,:final].+N,:label];

# ╔═╡ 8cf64c29-4e99-4222-9bc6-0b658fcda34a
md"""
### Vega-Lite specifications with Julia
"""

# ╔═╡ 9a04b97a-8a74-4c21-b68d-0f3382b6f16d
p1 =@vlplot("data"=df,
	mark={"type"=:circle,"size"=200,"opacity"=1},
	selection={"grid"={
	"type"=:interval,
	"resolve"=:global,
	"bind"=:scales,
	"translate"="[mousedown[!event.shiftKey], window:mouseup] > window:mousemove!",
                "zoom"="wheel![!event.shiftKey]"}},
    x={:x1,"type"="quantitative"},
    y={:x2,"type"="quantitative"},
    color={:dataset, "type"="nominal"},"height"=500,"width"=500);

# ╔═╡ 36accf29-47e4-445f-89da-c43cd49628fc
p2 = @vlplot(data=df,
    mark={type=:image, width=20,height=20},
    x={:x1,type="quantitative","axis"={"grid"=true}},
    y={:x2,type="quantitative"},
    selection={grid={
                type=:interval,
                resolve=:global,
                bind=:scales,
                translate="[mousedown[!event.shiftKey], window:mouseup] > window:mousemove!",
                zoom="wheel![!event.shiftKey]"}
    },
    url ={field=:img, type="nominal"}
    ,height=500,width=500);

# ╔═╡ 8be7896a-bb7b-4f28-a517-090094c26af5
c1 = @vlplot("data"=source,"height"=500,"width"=500,
    "mark"={:rect},
    "x"={"field"=:label,"type"="ordinal","sort"="ascending",
		"axis"={"orient"="top","labelAngle"=0}},
    "y"={"field"=:fmnist_label,"type"="ordinal","sort"="ascending"},
    "color"={"field"=:label,aggregate="count"},
    "config"= {"axis"= {"grid"= true, "tickBand"= "extent"}}
);

# ╔═╡ 8feb1c33-ba6d-449a-8676-b1144d4d4312
md"""
### Plotting with VegaLite directly from JavaScript
Plot below is the 2D projection of both MNIST and FashionMNIST using
UMAP. An Optimal Transport between the datasets is calculated using the
`OptimalTranposrt.jl` package. This plot is one of the "key" visualizations, although there isn't yet any interactivity.
In the final project, the user will be able to select datapoints, visulize information, perform augumentations to the dataset and understand how this can improve the Transfer Learning capability between the models.
"""

# ╔═╡ 3ce0657e-5487-43c9-a28c-7661c95a1486
md"""
This is another key visualization. It's a Heatmap showing how the labels are being transfered among the datasets. For example, note that the MNIST label "0" is being transfered almost exclusively to the FashionMNIST label "1". Hence, this implies that when doing the trasnfer learning, the model trained on MNIST can perform well on classifying "1" on the FashionMNIST. In contrast, the label "9" is very spread out among different labels, which can indicate that perhaps some data augumentation might improve the transferability. Another aspect that will be studied is the effect of label imbalance.
"""

# ╔═╡ eaeff487-d56a-4806-b1e9-63e2e9c8ba5e
@htl("""
	<head>
    <title>Embedding Vega-Lite</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
  </head>
  <body>
    <div id="vis2"></div>

    <script type="text/javascript">
	const spec = JSON.parse($(json(c1)));
  	vegaEmbed("#vis2", spec)
	.then(result => console.log(result))
      .catch(console.warn);
    </script>
  </body>
""")

# ╔═╡ 742ef2ec-4c23-46e7-ad39-ff838ef156b1
md"""
### Using D3 with Pluto
This will allow to create more interactivity.
Still on progress...
"""

# ╔═╡ 839f0087-5890-462d-8507-70b3c3db797d
@htl("""
		
<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>

<script id="hello">

const positions = JSON.parse($(dfjson))
	
const svg = this == null ? DOM.svg(600,500) : this
const s = this == null ? d3.select(svg) : this.s

s.selectAll("circle")
	.data(positions)
	.join("circle")
    .transition()
    .duration(300)
	.attr("cx", d => d.x1*50+300)
	.attr("cy", d => d.x2*50+200)
	.attr("r", 10)
	.attr("fill", "gray")


const output = svg
output.s = s
return output
</script>

""")

# ╔═╡ 11238559-ee1f-4000-91f3-a8fda2947393
load(img_url[1])

# ╔═╡ 835d761d-bfe5-45f6-919d-d0c03711a5c8
md"""
### Auxiliary Functions
"""

# ╔═╡ a4f33a01-dbb1-4f68-ae20-1d71262514b5
function CreateEdges(μ,ν,γ)
    edges = Array{Float64}(undef, 0, 2)
    pe    = []
    for i in 1:size(μ)[1], j in 1:size(ν)[1]
        edges  = vcat(edges,[μ[i,1],μ[i,2]]')
        edges  = vcat(edges,[ν[j,1],ν[j,2]]')
        pe     = vcat(pe,string([i,j]))
        pe     = vcat(pe,string([i,j]))
    end
    df = DataFrame(edges_x=edges[:,1],edges_y = edges[:,2],pe=pe);
    edge_w = []
    for i in 1:size(γ)[1], j in 1:size(γ)[1]
        edge_w = vcat(edge_w,γ[i,j])
        edge_w = vcat(edge_w,γ[i,j])
    end
    df[!,"ew"] = edge_w./maximum(edge_w);
    return df
end

# ╔═╡ fa1e471f-6b53-46c3-8028-f2aada534378
μ = res_jl[:,1:N]';

# ╔═╡ 76dcba78-d8ea-4544-971f-22c65d8042ef
ν = res_jl[:,N+1:2*N]';

# ╔═╡ f2db848c-1467-43d6-a569-d20859511668
C = pairwise(SqEuclidean(), μ', ν');

# ╔═╡ dcf59916-30dc-4a88-9840-a32df5ec0010
γ = sinkhorn(ones(N)/N,ones(N)/N,C,1);

# ╔═╡ 894c3e2b-6298-4ae8-ae80-843257205c95
filter = 0.3;

# ╔═╡ c18d6f95-3bfc-4492-95a2-ed3ab07d471e
edg = CreateEdges(μ,ν,γ);

# ╔═╡ 6c47e454-2a29-4114-8fdc-738b3cfe1427
edges = edg[edg[:,:ew] .>= filter,:];

# ╔═╡ 78cb1ec7-489b-4ed6-83d5-0ac79cc489bf
e1 = @vlplot(
        "mark"={"type"=:line,"color"="black","clip"=false},
        "data" = edges,
        "encoding"={
        "x"={"edges_x:q","axis"=nothing},
        "y"={"edges_y:q","axis"=nothing},
        "opacity"={"ew:q","legend"=nothing},
        "size"={"ew:o","scale"={"range"=[0,2]},"legend"=nothing},
        "detail"={"pe:o"}}
    );

# ╔═╡ bc869612-6e0f-43f1-9b8c-64558fdf7ea7
v1 = @vlplot("view"={stroke=nothing})+e1+p1;

# ╔═╡ a7db774c-363b-4a92-9c20-df4477c4a135
@htl("""
	<head>
    <title>Embedding Vega-Lite</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
  </head>
  <body>
    <div id="vis"></div>

    <script type="text/javascript">
	const spec = JSON.parse($(json(v1)));
  	vegaEmbed("#vis", spec)
	.then(result => console.log(result))
      .catch(console.warn);
    </script>
  </body>
""")

## `@htl` is a `macro` in Julia. A macro is a function written to perform meta-programming. The `@htl` macro is from the `HypertextLiteral.jl` package, and it parses the string in julia to `html`.

# ╔═╡ 7afdf3e1-89f6-432b-b628-a71e78da1366
f(x) = argmax(γ[x,:])

# ╔═╡ 62e6baa1-b46c-40bf-85b6-1982506ce6f0
g(x) = argmax(γ[:,x])

# ╔═╡ b11825b7-062d-4d9c-b941-1af4128b513f
mnistorigin = collect(1:N)

# ╔═╡ e9624341-1426-4a8c-902d-a4739e925c99
fmnistorigin = collect(1:N)


# ╔═╡ 46a93d19-5dd8-41cb-92a7-f9bddef187a2
mnistfinal = f.(mnistorigin);

# ╔═╡ 30123390-64fd-47c1-932b-adef269724fa
fmnistfinal = g.(fmnistorigin);

# ╔═╡ b9b87cca-feeb-42ea-911b-83f1a9339194
df[!,:origin] = vcat(mnistorigin,fmnistorigin);

# ╔═╡ fc45b031-97a8-413b-8e5c-c05cae5f4dcb
df[!,:final] = vcat(mnistfinal,fmnistfinal);

# ╔═╡ c6e21e55-1bb0-4d51-873a-39ce688129b9


# ╔═╡ 2cb1246c-97b7-473d-99d5-580e9e1d85eb


# ╔═╡ 07b99ed2-ee87-4cea-bbbf-f4918dc507d2


# ╔═╡ Cell order:
# ╟─2ffddf10-bd51-11eb-12cb-f1add38b47fb
# ╟─b3a49e8b-b54c-4247-8370-c2a917e57056
# ╠═8529c382-f72f-44f7-8ccd-ce68ab03776e
# ╠═4ff7b77e-27e7-4034-8939-07cc319712ad
# ╠═73b2d14c-1f89-425c-92a0-57591f21e8fd
# ╠═f411e2a4-66b2-41e0-90c0-a948cd6789f0
# ╠═f2c0afcc-016a-4802-b75b-80d258b45af6
# ╟─b3fd7749-18ef-4033-9e2d-431ee284c11b
# ╟─4a741e16-7e80-43cb-bfe3-63ae442d61f1
# ╠═dc2c4b08-9c1c-4555-bf7b-60be8e6bd0d9
# ╟─75b36234-7d5b-4527-9274-2239046b556a
# ╠═068369ca-a6db-4f01-b192-1256332202f0
# ╟─b89edb81-2e62-4b2a-8ce3-3e4c25a31b55
# ╠═23c1d16f-d23c-4a99-b4c9-db3ad782e8fc
# ╠═df0f24fd-f847-40fb-b3dc-12350face55f
# ╟─29ac65a4-a1c1-47a8-a691-be90f988709f
# ╠═3994768a-526e-4116-8dee-f398c7a36ffd
# ╟─21b3b741-1ea1-49a4-a6ae-b22666f53e19
# ╠═92af2802-acb2-4b67-9449-1fe793174df7
# ╠═32a4e97d-ecf0-4ff0-9d2d-0d6f47b855f2
# ╟─8cf64c29-4e99-4222-9bc6-0b658fcda34a
# ╠═9a04b97a-8a74-4c21-b68d-0f3382b6f16d
# ╠═36accf29-47e4-445f-89da-c43cd49628fc
# ╠═8be7896a-bb7b-4f28-a517-090094c26af5
# ╠═78cb1ec7-489b-4ed6-83d5-0ac79cc489bf
# ╠═bc869612-6e0f-43f1-9b8c-64558fdf7ea7
# ╟─8feb1c33-ba6d-449a-8676-b1144d4d4312
# ╠═a7db774c-363b-4a92-9c20-df4477c4a135
# ╟─3ce0657e-5487-43c9-a28c-7661c95a1486
# ╠═eaeff487-d56a-4806-b1e9-63e2e9c8ba5e
# ╟─742ef2ec-4c23-46e7-ad39-ff838ef156b1
# ╠═839f0087-5890-462d-8507-70b3c3db797d
# ╠═11238559-ee1f-4000-91f3-a8fda2947393
# ╟─835d761d-bfe5-45f6-919d-d0c03711a5c8
# ╠═a4f33a01-dbb1-4f68-ae20-1d71262514b5
# ╠═fa1e471f-6b53-46c3-8028-f2aada534378
# ╠═76dcba78-d8ea-4544-971f-22c65d8042ef
# ╠═f2db848c-1467-43d6-a569-d20859511668
# ╠═dcf59916-30dc-4a88-9840-a32df5ec0010
# ╠═894c3e2b-6298-4ae8-ae80-843257205c95
# ╠═c18d6f95-3bfc-4492-95a2-ed3ab07d471e
# ╠═6c47e454-2a29-4114-8fdc-738b3cfe1427
# ╠═7afdf3e1-89f6-432b-b628-a71e78da1366
# ╠═62e6baa1-b46c-40bf-85b6-1982506ce6f0
# ╠═b11825b7-062d-4d9c-b941-1af4128b513f
# ╠═e9624341-1426-4a8c-902d-a4739e925c99
# ╠═46a93d19-5dd8-41cb-92a7-f9bddef187a2
# ╠═30123390-64fd-47c1-932b-adef269724fa
# ╠═b9b87cca-feeb-42ea-911b-83f1a9339194
# ╠═fc45b031-97a8-413b-8e5c-c05cae5f4dcb
# ╠═c6e21e55-1bb0-4d51-873a-39ce688129b9
# ╠═2cb1246c-97b7-473d-99d5-580e9e1d85eb
# ╠═07b99ed2-ee87-4cea-bbbf-f4918dc507d2
