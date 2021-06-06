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
		Pkg.PackageSpec(name="MLDatasets"),
		Pkg.PackageSpec(name="VegaLite"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="Distances"),
		Pkg.PackageSpec(name="LinearAlgebra"),
		Pkg.PackageSpec(name="JSONTables"),
		Pkg.PackageSpec(name="Images")
    ])
	Pkg.add(url="https://github.com/JuliaOptimalTransport/OptimalTransport.jl")
	Pkg.add(url="https://github.com/davibarreira/LsqFit.jl")
	Pkg.add(url="https://github.com/davibarreira/UMAP.jl")
	
    using MLDatasets, VegaLite, DataFrames, Distances, LinearAlgebra, PlutoUI, HypertextLiteral, JSON, JSONTables, Images, OptimalTransport, UMAP
end

# ╔═╡ 2ffddf10-bd51-11eb-12cb-f1add38b47fb
md"""
# Dataset Transferability Analysis
A visual tool for improving transfer learning via data augumentation and Optimal Transport
"""

# ╔═╡ b3a49e8b-b54c-4247-8370-c2a917e57056
md"""
### Installing and Importing Packages and Data
"""

# ╔═╡ e5494bbe-ce7d-4a63-8b9f-c0989b3acffb
begin
	mnist_x = reshape(MNIST.traintensor(Float64),28*28,:);
	mnist_y = MNIST.trainlabels(1:size(mnist_x, 2));
	fmnist_x = reshape(FashionMNIST.traintensor(Float64),28*28,:);
	fmnist_y = FashionMNIST.trainlabels(1:size(fmnist_x, 2));

	N = 100;
	mnist_x  = mnist_x'[1:N,:];
	mnist_y  = mnist_y[1:N];
	fmnist_x = fmnist_x'[1:N,:];
	fmnist_y = fmnist_y[1:N];
	img_url = vcat(
					["./images/mnist_"*string(i)*".png" for i in 1:N],
					["./images/fmnist_"*string(i)*".png" for i in 1:N]);
end

# ╔═╡ b3fd7749-18ef-4033-9e2d-431ee284c11b
md"""
### Data Wrangling
"""

# ╔═╡ 4a741e16-7e80-43cb-bfe3-63ae442d61f1
md"""
Sample size for testing 
"""

# ╔═╡ 75b36234-7d5b-4527-9274-2239046b556a
md"""
Applying UMAP to reduce dimensionality of the datasets
"""

# ╔═╡ 068369ca-a6db-4f01-b192-1256332202f0
res = umap(hcat(mnist_x',fmnist_x'); n_neighbors=10, min_dist=0.001, n_epochs=200)';

# ╔═╡ b89edb81-2e62-4b2a-8ce3-3e4c25a31b55
md"""
Generating dataset for plots
"""

# ╔═╡ df0f24fd-f847-40fb-b3dc-12350face55f
begin
	df = DataFrame(
		x     = res[:,1],
		y     = res[:,2],
		img    = img_url,
		label  = vcat(mnist_y[1:N],fmnist_y[1:N]),
		dataset= vcat(["mnist" for i in 1:N],["fmnist" for i in 1:N]));

end

# ╔═╡ 29ac65a4-a1c1-47a8-a691-be90f988709f
md"""
Dataframe to Json to pass to JavaScript
"""

# ╔═╡ 3994768a-526e-4116-8dee-f398c7a36ffd
dfjson = arraytable(df)

# ╔═╡ 21b3b741-1ea1-49a4-a6ae-b22666f53e19
md"""
Dataset for heatmap
"""

# ╔═╡ 92af2802-acb2-4b67-9449-1fe793174df7
source = df[1:N,:];

# ╔═╡ 32a4e97d-ecf0-4ff0-9d2d-0d6f47b855f2
# source[!,:fmnist_label] = df[source[!,:final].+N,:label];

# ╔═╡ 8cf64c29-4e99-4222-9bc6-0b658fcda34a
md"""
### Vega-Lite specifications with Julia
"""

# ╔═╡ 9a04b97a-8a74-4c21-b68d-0f3382b6f16d
v1 =@vlplot("data"=df,
	"mark"={"type"=:circle,"size"=200,"opacity"=1},
	"selection"={"grid"={
	"type"=:interval,
	"resolve"=:global,
	"bind"=:scales,
	"translate"="[mousedown[!event.shiftKey], window:mouseup] > window:mousemove!",
                "zoom"="wheel![!event.shiftKey]"}},
    x={:x,"type"="quantitative"},
    y={:y,"type"="quantitative"},
    color={:dataset, "type"="nominal"},"height"=500,"width"=500);

# ╔═╡ 8feb1c33-ba6d-449a-8676-b1144d4d4312
md"""
### Plotting with VegaLite directly from JavaScript
Plot below is the 2D projection of both MNIST and FashionMNIST using
UMAP. An Optimal Transport between the datasets is calculated using the
`OptimalTranposrt.jl` package. This plot is one of the "key" visualizations, although there isn't yet any interactivity.
In the final project, the user will be able to select datapoints, visulize information, perform augumentations to the dataset and understand how this can improve the Transfer Learning capability between the models.
"""

# ╔═╡ a7db774c-363b-4a92-9c20-df4477c4a135
# @htl("""
# 	<head>
#     <title>Embedding Vega-Lite</title>
#     <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
#   </head>
#   <body>
#     <div id="vis"></div>

#     <script type="text/javascript">
# 	const spec = JSON.parse($(json(v1)));
#   	vegaEmbed("#vis", spec)
# 	.then(result => console.log(result))
#       .catch(console.warn);
#     </script>
#   </body>
# """)

## `@htl` is a `macro` in Julia. A macro is a function written to perform meta-programming. The `@htl` macro is from the `HypertextLiteral.jl` package, and it parses the string in julia to `html`.

# ╔═╡ 3ce0657e-5487-43c9-a28c-7661c95a1486
md"""
This is another key visualization. It's a Heatmap showing how the labels are being transfered among the datasets. For example, note that the MNIST label "0" is being transfered almost exclusively to the FashionMNIST label "1". Hence, this implies that when doing the trasnfer learning, the model trained on MNIST can perform well on classifying "1" on the FashionMNIST. In contrast, the label "9" is very spread out among different labels, which can indicate that perhaps some data augumentation might improve the transferability. Another aspect that will be studied is the effect of label imbalance.
"""

# ╔═╡ c1c693c0-1c57-43c8-af20-9cd5e9c7d6af
@htl("""
<head>
    <title>Embedding Vega-Lite</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>

  <body>
    
	<div id="vis"></div>
    <script type="text/javascript">
	const spec = JSON.parse($(json(v1)));
  	vegaEmbed("#vis", spec,{renderer: "svg"})
	.then(() =>{
		const svg = d3.select("#vis").selectAll("svg")
	
		const dot = svg.selectAll(".mark-symbol.role-mark")
			.selectAll("path")
			.attr("fill","green")
	console.log(svg)
		
		const brush = d3.brush()
      	.on("start brush end", brushed);


	
	
	
	}
	)
      .catch(console.warn);
    </script>
	
	<script id="d3">
		

	</script>
	
	
  </body>
""")

# ╔═╡ 742ef2ec-4c23-46e7-ad39-ff838ef156b1
md"""
### Using D3 with Pluto
This will allow to create more interactivity.
Still on progress...
"""

# ╔═╡ 7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
@htl("""
	<head>
    <title>Embedding Vega-Lite</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
  </head>
  <body>
    <div id="myvis"></div>

    <script id="createplot">
		var height = 400;
		var width = 600;
		var margin = ({top: 20, right: 30, bottom: 30, left: 40});
	
		const data = JSON.parse($(dfjson))
	
		const svg = d3
			.select("#myvis")
			.append("svg")
			.attr("width", width + margin.left + margin.right)
    		.attr("height", height + margin.top + margin.bottom)
	
	const x = d3.scaleLinear().domain(d3.extent(data, d => d.x)).nice()
    .range([margin.left, width - margin.right]);


const y = d3.scaleLinear()
   .domain(d3.extent(data, d => d.y)).nice()
   .range([height - margin.bottom, margin.top]);
	
	const dot = svg.append("g")
	.selectAll("circle")
	.data(data)
	.join("circle")
    .transition()
    .duration(300)
	.attr("cx", d => x(d.x))
	.attr("cy", d => y(d.y))
	.attr("r", 5)
	.attr("fill", "steelblue")
    .attr("stroke", "steelblue")
	.attr("stroke-width", 2)
	.attr('opacity',0.5)
	
	function brushed({selection}) {
    let value = [];
    if (selection) {
      const [[x0, y0], [x1, y1]] = selection;
      value = dot
        .style("stroke", "gray")
        .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
        .style("stroke", "steelblue")
        .data();
    } else {
      dot.style("stroke", "steelblue");
    }
    svg.property("value", value).dispatch("input");
  }
	const brush = d3.brush()
      .on("start brush end", brushed);
	
	 svg.call(brush);
	
    </script>
  </body>
""")

# ╔═╡ 839f0087-5890-462d-8507-70b3c3db797d
@htl("""
		
<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>

<script id="hello">
	
var height = 400;
var width = 600;
var margin = ({top: 20, right: 30, bottom: 30, left: 40});


const data = JSON.parse($(dfjson))

const x = d3.scaleLinear().domain(d3.extent(data, d => d.x)).nice()
    .range([margin.left, width - margin.right]);


const y = d3.scaleLinear()
   .domain(d3.extent(data, d => d.y)).nice()
   .range([height - margin.bottom, margin.top]);

const svg = DOM.svg(width,height)
const dot = d3.select(svg)


dot.selectAll("circle")
	.data(data)
	.join("circle")
    .transition()
    .duration(300)
	.attr("cx", d => x(d.x))
	.attr("cy", d => y(d.y))
	.attr("r", 5)
	.attr("fill", "steelblue")
    .attr("stroke", "steelblue")
	.attr("stroke-width", 2)
	.attr('opacity',0.5)


  function brushed({selection}) {
    let value = [];
    if (selection) {
      const [[x0, y0], [x1, y1]] = selection;
      value = dot
        .style("stroke", "gray")
        .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
        .style("stroke", "steelblue")
        .data();
    } else {
      dot.style("stroke", "steelblue");
    }
    svg.property("value", value).dispatch("input");
  }

const brush = d3.brush()
      .on("start brush end", brushed);
// svg.call(brush);
	
const output = svg
output.dot = dot
return output
</script>

""")

# ╔═╡ 07120a08-226b-4907-87c7-f5d63af616a7
# function brushed({selection}) {
#     let value = [];
#     if (selection) {
#       const [[x0, y0], [x1, y1]] = selection;
#       value = dot
#         .style("stroke", "gray")
#         .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
#         .style("stroke", "steelblue")
#         .data();
#     } else {
#       dot.style("stroke", "steelblue");
#     }
#     svg.property("value", value).dispatch("input");
# }

# const brush = d3.brush()
#       .on("start brush end", brushed);
	
# svg.call(brush);

# ╔═╡ 835d761d-bfe5-45f6-919d-d0c03711a5c8
md"""
### Auxiliary Functions
"""

# ╔═╡ 70a5b623-418e-4b91-a1b2-dd88a26d5756
res_jl = umap(hcat(mnist_x[:,1:N],fmnist_x[:,1:N]); n_neighbors=10, min_dist=0.001, n_epochs=200);

# ╔═╡ Cell order:
# ╟─2ffddf10-bd51-11eb-12cb-f1add38b47fb
# ╟─b3a49e8b-b54c-4247-8370-c2a917e57056
# ╠═8529c382-f72f-44f7-8ccd-ce68ab03776e
# ╠═e5494bbe-ce7d-4a63-8b9f-c0989b3acffb
# ╟─b3fd7749-18ef-4033-9e2d-431ee284c11b
# ╟─4a741e16-7e80-43cb-bfe3-63ae442d61f1
# ╟─75b36234-7d5b-4527-9274-2239046b556a
# ╠═068369ca-a6db-4f01-b192-1256332202f0
# ╟─b89edb81-2e62-4b2a-8ce3-3e4c25a31b55
# ╠═df0f24fd-f847-40fb-b3dc-12350face55f
# ╟─29ac65a4-a1c1-47a8-a691-be90f988709f
# ╠═3994768a-526e-4116-8dee-f398c7a36ffd
# ╟─21b3b741-1ea1-49a4-a6ae-b22666f53e19
# ╠═92af2802-acb2-4b67-9449-1fe793174df7
# ╠═32a4e97d-ecf0-4ff0-9d2d-0d6f47b855f2
# ╟─8cf64c29-4e99-4222-9bc6-0b658fcda34a
# ╠═9a04b97a-8a74-4c21-b68d-0f3382b6f16d
# ╟─8feb1c33-ba6d-449a-8676-b1144d4d4312
# ╠═a7db774c-363b-4a92-9c20-df4477c4a135
# ╟─3ce0657e-5487-43c9-a28c-7661c95a1486
# ╠═c1c693c0-1c57-43c8-af20-9cd5e9c7d6af
# ╟─742ef2ec-4c23-46e7-ad39-ff838ef156b1
# ╠═7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
# ╠═839f0087-5890-462d-8507-70b3c3db797d
# ╠═07120a08-226b-4907-87c7-f5d63af616a7
# ╟─835d761d-bfe5-45f6-919d-d0c03711a5c8
# ╠═70a5b623-418e-4b91-a1b2-dd88a26d5756
