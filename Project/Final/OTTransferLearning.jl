### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

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

# ╔═╡ 1fbf4fa5-3eb5-4866-9b55-4bf9c5967250
include("otdd.jl")

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
					["http://localhost:5000/mnist_"*string(i)*".png" for i in 1:N],
					["http://localhost:5000/fmnist_"*string(i)*".png" for i in 1:N]);
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
		img = img_url,
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

# ╔═╡ 3ce0657e-5487-43c9-a28c-7661c95a1486
md"""
This is another key visualization. It's a Heatmap showing how the labels are being transfered among the datasets. For example, note that the MNIST label "0" is being transfered almost exclusively to the FashionMNIST label "1". Hence, this implies that when doing the trasnfer learning, the model trained on MNIST can perform well on classifying "1" on the FashionMNIST. In contrast, the label "9" is very spread out among different labels, which can indicate that perhaps some data augumentation might improve the transferability. Another aspect that will be studied is the effect of label imbalance.
"""

# ╔═╡ c1c693c0-1c57-43c8-af20-9cd5e9c7d6af
# @htl("""
# <head>
#     <title>Embedding Vega-Lite</title>
#     <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
# 	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>

#   <body>
    
# 	<div id="vis"></div>
#     <script type="text/javascript">
# 	const spec = JSON.parse($(json(v1)));
#   	vegaEmbed("#vis", spec,{renderer: "svg"})
# 	.then(() =>{
# 		const svg = d3.select("#vis").selectAll("svg")
	
# 		const dot = svg.selectAll(".mark-symbol.role-mark")
# 			.selectAll("path")
# 			.attr("fill","green")
# 	console.log(svg)
		
# 		const brush = d3.brush()
#       	.on("start brush end", brushed);


	
	
	
# 	}
# 	)
#       .catch(console.warn);
#     </script>
	
# 	<script id="d3">
		

# 	</script>
	
	
#   </body>
# """)

# ╔═╡ 742ef2ec-4c23-46e7-ad39-ff838ef156b1
md"""
### Using D3 with Pluto
This will allow to create more interactivity.
Still on progress...
"""

# ╔═╡ 7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
Scatter = @htl("""
	<div>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
	<script>
	let cell = currentScript.closest("pluto-cell")
	cell.style.width = "1000px"
	</script>
    <div id="myvis"></div>

    <script id="createplot">
	var div = currentScript.parentElement
	
	var selection = 0;
	
		var height = 500;
		var width = 800;
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
	
	const color = d3.scaleOrdinal()
    .domain(["mnist", "fmnist"])
    .range([ "#440154ff", "#21908dff"]);
	
	const dot = svg.append("g")
	.selectAll("circle")
	.data(data)
	.join("circle")
	.attr("cx", d => x(d.x))
	.attr("cy", d => y(d.y))
	.attr("r", 1)
	.attr("fill", "steelblue")
    .attr("stroke", "steelblue")
	.attr("stroke-width", 0)
	.attr('opacity',0.5)
	
var myimage = svg.selectAll('image')
	.data(data)
	.join('image')
    .attr('xlink:href', d => d.img)
	.attr("x", d => x(d.x))
	.attr("y", d => y(d.y))
    .attr('width', 30)
    .attr('height', 30)
	.attr('opacity',1)


	function brushed({selection}) {
	let value = [];
    if (selection) {
      const [[x0, y0], [x1, y1]] = selection;
      value = dot
        .style("fill", "gray")
		.attr("class","unselected")
        .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
        .style("fill", "steelblue")
		.attr("class","selected")
        .data();
	    
      myimage.attr('opacity',0.3)
		.attr("class","unselected")
        .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
		.attr("class","selected")
		.attr('opacity',1.0);
    } else {
      dot.style("fill", "steelblue")
		 .attr("class","selected");
	  myimage.attr("class","selected").attr('opacity',1.0);
    }
	div.value = value;
    svg.property("value", value).dispatch("input");
  }
	const brush = d3.brush()
      .on("start brush end", brushed);
	
	
	svg.call(d3.zoom()
      	.extent([[0, 0], [width, height]])
		.translateExtent([[0, 0], [width, height]])
      	.scaleExtent([1, 8])
      	.on("zoom", zoomed)).on("touchstart.zoom", null).on("mousedown.zoom", null)
		.on("dblclick.zoom", null);

  function zoomed({transform}) {
    myimage.attr("transform", transform);
  }
	svg.call(brush);

	
	div.value = svg.selectAll("selected")
    </script>
	</div>
""")

# ╔═╡ 839f0087-5890-462d-8507-70b3c3db797d
GetSelected(text="Get Selection") = @htl("""
	<div id="ok">
	<button>$(text)</button>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
    <script id="selection">
	var div = currentScript.parentElement
	var button = div.querySelector("button")
	button.addEventListener("click", (e) => {
		div.value = d3.select("#myvis").selectAll("svg").selectAll(".selected").data()
		div.dispatchEvent(new CustomEvent("input"))
		e.preventDefault()
	})
	const svg = d3.select("#myvis").selectAll("svg").selectAll(".selected")
	div.value = svg.data()
    </script>
	</div>
""")

# ╔═╡ a9cb0024-ae23-4fc9-81d8-4ea335884900
@bind selected GetSelected()

# ╔═╡ 07120a08-226b-4907-87c7-f5d63af616a7
selected

# ╔═╡ d468ee56-e522-421b-91b3-66135b0e8683
@htl("""
	<script>
	var ok = "ok";
	</script>
	""")

# ╔═╡ 835d761d-bfe5-45f6-919d-d0c03711a5c8
md"""
### Auxiliary Functions
"""

# ╔═╡ 530bc707-21f7-451f-9fee-6b3430759e0e
	# var rec = svg.selectAll('rect')
	# .data(data)
	# .join('rect')
	# .attr("x", d => x(d.x))
	# .attr("y", d => y(d.y))
	# .attr('width', 30)
	# .attr('height', 30)
	# .style("fill", function(d) { return color(d.dataset)} )
	# .attr('opacity',0.2)

# ╔═╡ 70a5b623-418e-4b91-a1b2-dd88a26d5756
res_jl = umap(hcat(mnist_x[:,1:N],fmnist_x[:,1:N]); n_neighbors=10, min_dist=0.001, n_epochs=200);

# ╔═╡ e526398c-e77e-43fe-bfb4-638ff2ad579b


# ╔═╡ 6c333ac9-22a6-4353-a4f1-67bebdaadc24


# ╔═╡ 1b066cf7-bf1b-4445-9e58-275679838973
@htl("""
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
	<script>
	let cell = currentScript.closest("pluto-cell")
	cell.style.width = "1000px"
	</script>
    <div id="myvis2"></div>

    <script id="createplot">
		var height = 500;
		var width = 800;
		var margin = ({top: 20, right: 30, bottom: 30, left: 40});
	
		const data = JSON.parse($(dfjson))
	
		const svg = d3
			.select("#myvis2")
			.append("svg")
			.attr("width", width + margin.left + margin.right)
    		.attr("height", height + margin.top + margin.bottom)
	
	
	const x = d3.scaleLinear().domain(d3.extent(data, d => d.x)).nice()
    .range([margin.left, width - margin.right]);

	const y = d3.scaleLinear()
	   .domain(d3.extent(data, d => d.y)).nice()
	   .range([height - margin.bottom, margin.top]);
	
	const color = d3.scaleOrdinal()
    .domain(["mnist", "fmnist"])
    .range([ "#440154ff", "#21908dff"]);
	
	const dot = svg.append("g")
	.selectAll("circle")
	.data(data)
	.join("circle")
	.attr("cx", d => x(d.x))
	.attr("cy", d => y(d.y))
	.attr("r", 1)
	.attr("fill", "steelblue")
    .attr("stroke", "steelblue")
	.attr("stroke-width", 0)
	.attr('opacity',0.5)
	
	 svg.call(d3.zoom()
      .extent([[0, 0], [width, height]])
      .scaleExtent([1, 8])
      .on("zoom", zoomed));

  function zoomed({transform}) {
    dot.attr("transform", transform);
  }
	
    </script>
""")

# ╔═╡ cc3be0be-41e0-497c-9186-875459eead51


# ╔═╡ 6e943451-c831-4a65-99b5-65fbd6ee1753


# ╔═╡ e44468be-2f1d-478d-a4bb-7009fefe1098


# ╔═╡ 1d9b2a30-3b5f-47e2-bd11-af831169cda0


# ╔═╡ 5db078b7-a85d-46b6-b533-d2c05117b9d3


# ╔═╡ 5e63314f-bd6d-4486-b82f-dbad5615e63a


# ╔═╡ Cell order:
# ╟─2ffddf10-bd51-11eb-12cb-f1add38b47fb
# ╟─b3a49e8b-b54c-4247-8370-c2a917e57056
# ╠═8529c382-f72f-44f7-8ccd-ce68ab03776e
# ╠═1fbf4fa5-3eb5-4866-9b55-4bf9c5967250
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
# ╟─8cf64c29-4e99-4222-9bc6-0b658fcda34a
# ╠═9a04b97a-8a74-4c21-b68d-0f3382b6f16d
# ╟─8feb1c33-ba6d-449a-8676-b1144d4d4312
# ╟─3ce0657e-5487-43c9-a28c-7661c95a1486
# ╟─c1c693c0-1c57-43c8-af20-9cd5e9c7d6af
# ╟─742ef2ec-4c23-46e7-ad39-ff838ef156b1
# ╠═7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
# ╟─839f0087-5890-462d-8507-70b3c3db797d
# ╟─a9cb0024-ae23-4fc9-81d8-4ea335884900
# ╠═07120a08-226b-4907-87c7-f5d63af616a7
# ╠═d468ee56-e522-421b-91b3-66135b0e8683
# ╟─835d761d-bfe5-45f6-919d-d0c03711a5c8
# ╠═530bc707-21f7-451f-9fee-6b3430759e0e
# ╠═70a5b623-418e-4b91-a1b2-dd88a26d5756
# ╠═e526398c-e77e-43fe-bfb4-638ff2ad579b
# ╠═6c333ac9-22a6-4353-a4f1-67bebdaadc24
# ╠═1b066cf7-bf1b-4445-9e58-275679838973
# ╠═cc3be0be-41e0-497c-9186-875459eead51
# ╠═6e943451-c831-4a65-99b5-65fbd6ee1753
# ╠═e44468be-2f1d-478d-a4bb-7009fefe1098
# ╠═1d9b2a30-3b5f-47e2-bd11-af831169cda0
# ╠═5db078b7-a85d-46b6-b533-d2c05117b9d3
# ╠═5e63314f-bd6d-4486-b82f-dbad5615e63a
