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
		Pkg.PackageSpec(name="Images"),
		Pkg.PackageSpec(name="ImageContrastAdjustment"),
		Pkg.PackageSpec(name="ImageCore"),	
		Pkg.PackageSpec(name="ImageTransformations"),
		Pkg.PackageSpec(name="Rotations"),
		Pkg.PackageSpec(name="CoordinateTransformations")
    ])
	Pkg.add(url="https://github.com/JuliaOptimalTransport/OptimalTransport.jl")
	Pkg.add(url="https://github.com/davibarreira/LsqFit.jl")
	Pkg.add(url="https://github.com/davibarreira/UMAP.jl")
	
    using MLDatasets, VegaLite, DataFrames, Distances, LinearAlgebra, PlutoUI, HypertextLiteral, JSON, JSONTables, Images, OptimalTransport, UMAP, ImageContrastAdjustment, ImageCore, ImageTransformations, Rotations
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
		id    = collect(1:2*N),
		x     = res[:,1],
		y     = res[:,2],
		img = img_url,
		label  = vcat(mnist_y[1:N],fmnist_y[1:N]),
		dataset= vcat(["mnist" for i in 1:N],["fmnist" for i in 1:N]));
end;

# ╔═╡ c315a543-e9be-4b79-8449-b9175c923bb8
dfjson = arraytable(df)

# ╔═╡ 29ac65a4-a1c1-47a8-a691-be90f988709f
md"""
Dataframe to Json to pass to JavaScript
"""

# ╔═╡ 21b3b741-1ea1-49a4-a6ae-b22666f53e19
md"""
Dataset for heatmap
"""

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
        let cell = currentScript.closest("pluto-cell");
        cell.style.width = "1000px";
    </script>
    <div id="myvis"></div>

    <script id="createplot">
        var div = currentScript.parentElement;

        var selection = 0;

        var height = 300;
        var width = 460;
        var margin = { top: 20, right: 30, bottom: 30, left: 40 };

        const data = JSON.parse($(dfjson));

        const svg = d3
            .select("#myvis")
            .append("svg")
            .attr("width", width * 2 + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        svg.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("height", height + margin.top + margin.bottom)
            .attr("width", width + margin.left)
            .style("fill", "white")
            .attr("stroke", "grey");

        const x = d3
            .scaleLinear()
            .domain(d3.extent(data, (d) => d.x))
            .nice()
            .range([0, width - 2 * margin.right]);

        const y = d3
            .scaleLinear()
            .domain(d3.extent(data, (d) => d.y))
            .nice()
            .range([height - margin.bottom, margin.top]);

        const color = d3.scaleOrdinal().domain(["mnist", "fmnist"]).range(["#440154ff", "#21908dff"]);

        var myimage = svg
            .append("g")
            .selectAll("image")
            .data(data)
            .join("image")
            .attr("xlink:href", (d) => d.img)
            .attr("x", (d) => x(d.x))
            .attr("y", (d) => y(d.y))
            .attr("width", 20)
            .attr("height", 20)
            .attr("opacity", 1);

        function brushed({ selection }) {
            let value = [];
            if (selection) {
                const [[x0, y0], [x1, y1]] = selection;

                value = myimage
                    .attr("opacity", 0.3)
                    .attr("class", "unselected")
                    .filter((d) => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
                    .attr("class", "selected")
                    .attr("opacity", 1.0)
                    .data();

                var visible = plot2
                    .selectAll("image")
                    .attr("opacity", 0)
                    .attr("class", "unselected")
                    .filter((d) => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
                    .attr("class", "selected")
                    .attr("opacity", 1.0)
                    .data();

                const x3 = d3
                    .scaleLinear()
                    .domain(d3.extent(visible, (d) => d.x))
                    .nice()
                    .range([margin.left, width - margin.right]);
                const y2 = d3
                    .scaleLinear()
                    .domain(d3.extent(visible, (d) => d.y))
                    .nice()
                    .range([height - margin.bottom, margin.top]);
                plot2
                    .selectAll("image")
                    .attr("opacity", 0)
                    .attr("class", "unselected")
                    .filter((d) => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
                    .attr("class", "selected")
                    .attr("opacity", 1.0)
                    .data(visible)
                    .attr("xlink:href", (d) => d.img)
                    .attr("x", (d) => x3(d.x))
                    .attr("y", (d) => y2(d.y));
            } else {
                myimage.attr("class", "selected").attr("opacity", 1.0);
            }
            div.value = value;
            svg.property("value", value).dispatch("input");
        }
        const brush = d3.brush().on("start brush end", brushed);

        svg.call(brush);

	
	
	
	function brushselection({ selection }) {
            let value = [];
            if (selection) {
                const [[x0, y0], [x1, y1]] = selection;

                var visible = plot2
                    .selectAll("image")
                    .attr("opacity", 0)
                    .attr("class", "unselected")
                    .filter((d) => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
                    .attr("class", "selected")
                    .attr("opacity", 1.0)
                    .data();

            } else {
                myimage.attr("class", "selected").attr("opacity", 1.0);
            }
            div.value = value;
            svg.property("value", value).dispatch("input");
        }
	
	const brushselector = d3.brush().on("start brush end", brushselection);

        svg.call(brush);
	
	
	
	
        const x2 = d3
            .scaleLinear()
            .domain(d3.extent(data, (d) => d.x))
            .nice()
            .range([margin.left, width - margin.right]);

        const y2 = d3
            .scaleLinear()
            .domain(d3.extent(data, (d) => d.y))
            .nice()
            .range([height - margin.bottom, margin.top]);

        svg.append("rect")
            .attr("transform", `translate(\${width},0)`)
            .attr("x", 0)
            .attr("y", 0)
            .attr("height", height + margin.top + margin.bottom)
            .attr("width", width + margin.left)
            .style("fill", "#F2F3F4")
            .attr("stroke", "grey");

        var plot2 = svg.append("g").attr("transform", `translate(\${width},0)`).attr("class", "plot2");

        var plt = plot2
            .selectAll(".selected")
            .data(data)
            .join("image")
            .attr("xlink:href", (d) => d.img)
            .attr("x", (d) => x2(d.x))
            .attr("y", (d) => y2(d.y))
            .attr("width", 60)
            .attr("height", 60)
            .attr("opacity", 0);

        div.value = svg.selectAll("selected");
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

# ╔═╡ 8a0319c9-51ac-4e99-a4d9-1c07bee83381
MNIST.convert2image(mnist_x[2,:])

# ╔═╡ 7b20f2d5-a2c7-4705-9576-f39cf4ca03f5
# mosaicview(
# 	[MNIST.convert2image(fmnist_x[selection_fmnist[1],:]) MNIST.convert2image(fmnist_x[selection_fmnist[1],:])])

# ╔═╡ 3aed32f1-f15c-4672-8641-e52c9a7c7671
@bind transformations Select(["none","equalization", "gamma"])

# ╔═╡ b2aafd47-c5c1-4ada-8d48-bfea30292d20
@bind choosedataset Select(["mnist","fmnist"])

# ╔═╡ bf62c705-49cc-4545-8bdf-316a61c9a5c0
@bind savetransformation Button("Save Modifications")

# ╔═╡ f6259df8-8fee-48d0-b5ec-04e0dbde1250


# ╔═╡ d468ee56-e522-421b-91b3-66135b0e8683
begin
	if length(selected)>0
		modified = DataFrame(selected[1])
		push!(modified,selected[2:end]...)
		selection_mnist = modified[modified[!,:dataset] .== "mnist",:id]
		selection_fmnist = modified[modified[!,:dataset] .== "fmnist",:id]
		selectionjson = arraytable(modified)
	else
		selectionjson = []
	end
end;

# ╔═╡ 95063639-9e69-4bff-85e0-31e642be8a0a
SampleView = @htl("""
	<div>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
	<script>
	let cell = currentScript.closest("pluto-cell")
	cell.style.width = "1000px"
	</script>
    <div id="sampleview"></div>

    <script id="createplot">
	var div = currentScript.parentElement
	
	var selection = 0;
	
		var height = 300;
		var width = 800;
		var margin = ({top: 20, right: 30, bottom: 30, left: 40});
	
		const data = JSON.parse($(selectionjson))
	
		const svg = d3
			.select("#sampleview")
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
	
	
var myimage = svg.selectAll('image')
	.data(data)
	.join('image')
    .attr('xlink:href', d => d.img)
	.attr("x", d => x(d.x))
	.attr("y", d => y(d.y))
    .attr('width', 50)
    .attr('height', 50)
	.attr('opacity',1)


	function brushed({selection}) {
	let value = [];
    if (selection) {
      const [[x0, y0], [x1, y1]] = selection;
	    
      value = myimage.attr('opacity',0.3)
		.attr("class","unselected")
        .filter(d => x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1)
		.attr("class","selected")
		.attr('opacity',1.0).data();
    } else {
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

# ╔═╡ 835d761d-bfe5-45f6-919d-d0c03711a5c8
md"""
### Auxiliary Functions
"""

# ╔═╡ 70a5b623-418e-4b91-a1b2-dd88a26d5756
res_jl = umap(hcat(mnist_x[:,1:N],fmnist_x[:,1:N]); n_neighbors=10, min_dist=0.001, n_epochs=200);

# ╔═╡ e526398c-e77e-43fe-bfb4-638ff2ad579b
function applyequalization(datasetarray, nbins=256)
    img  = Gray.(datasetarray)
    mimg = adjust_histogram(img, Equalization(nbins = nbins))
    return reshape(convert(Array{Float64}, mimg),28*28)
end


# ╔═╡ 583923b6-f08a-4074-94ff-2fc480e16277
function ApplyEqualization(img_ids,dataset)
    for id in img_ids
        if dataset == "mnist"
			mimg = applyequalization(mnist_x[id,:])
			mnist_x[id,:] = mimg
			df[id,:img] = "http://localhost:5000/modified/mnist_"* string(id)* ".png"
            save("./images/modified/mnist_"*string(id)*".png",MNIST.convert2image(mimg))
        else
			mimg = applyequalization(fmnist_x[id-N,:])
			fmnist_x[id-N,:] = mimg
			df[id,:img] = "http://localhost:5000/modified/fmnist_"* string(id)* ".png"
            save("./images/modified/fmnist_"*string(id)*".png",MNIST.convert2image(applyequalization(mimg)))
        end
    end
end

# ╔═╡ 6c333ac9-22a6-4353-a4f1-67bebdaadc24
function applygamma(datasetarray, gamma = 2)
    img  = Gray.(datasetarray)
    mimg = adjust_histogram(img, GammaCorrection(gamma = gamma))
    return reshape(convert(Array{Float64}, mimg),28*28)
end

# ╔═╡ 1150fee9-a9a5-4158-be90-eb72385cf3d1
let
	if length(selected)>0
		if choosedataset == "fmnist"
			if length(selection_fmnist) == 0
				"No selections"
			elseif transformations == "none"
				MNIST.convert2image(fmnist_x[selection_fmnist[1]-N,:])
			elseif transformations == "equalization"
				mimg = applyequalization(fmnist_x[selection_fmnist[1]-N,:],200)
				MNIST.convert2image(mimg)
			elseif transformations == "gamma"
				mimg = applygamma(fmnist_x[selection_fmnist[1]-N,:])
				MNIST.convert2image(mimg)
			end
		elseif choosedataset == "mnist"
			if length(selection_mnist) == 0
				"No selections"
			elseif transformations == "none"
				MNIST.convert2image(mnist_x[selection_mnist[1],:])
			elseif transformations == "equalization"
				mimg = applyequalization(mnist_x[selection_mnist[1],:],200)
				MNIST.convert2image(mimg)
			elseif transformations == "gamma"
				mimg = applygamma(mnist_x[selection_mnist[1],:])
				MNIST.convert2image(mimg)
			end
		end
	end
end

# ╔═╡ 57103a08-fefc-4c8c-85cb-a054de9edc5b
function ApplyGamma(img_ids,dataset)
    for id in img_ids
        if dataset == "mnist"
			mimg = applygamma(mnist_x[id,:])
			mnist_x[id,:] = mimg
			df[id,:img] = "http://localhost:5000/modified/mnist_"* string(id)* ".png"
            save("./images/modified/mnist_"*string(id)*".png",MNIST.convert2image(applygamma(mimg)))
        else
			mimg = applyequalization(fmnist_x[id-N,:])
			fmnist_x[id-N,:] = mimg
			df[id,:img] = "http://localhost:5000/modified/fmnist_"* string(id)* ".png"
            save("./images/modified/fmnist_"*string(id)*".png",MNIST.convert2image(applygamma(mimg)))
        end
    end
end

# ╔═╡ 589be23b-fc9b-4c5b-9316-64ce3074a281
let
	savetransformation
	if choosedataset == "fmnist"
		if transformations == "gamma"
			if length(selection_fmnist) > 0
				ApplyGamma(selection_fmnist, "fmnist")

			end
		elseif transformations == "equalization"
			if length(selection_fmnist) > 0
				ApplyEqualization(selection_fmnist, "fmnist")
			end
		end
	elseif choosedataset == "mnist"
		if transformations == "gamma"
			if length(selection_mnist) > 0
				ApplyGamma(selection_mnist, "mnist")
			end
		elseif transformations == "equalization"
			if length(selection_mnist) > 0
				ApplyEqualization(selection_mnist, "mnist")
			end
		end
	end
end

# ╔═╡ 1b066cf7-bf1b-4445-9e58-275679838973
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 1fbf4fa5-3eb5-4866-9b55-4bf9c5967250
# include("otdd.jl")
ot = ingredients("./otdd.jl");

# ╔═╡ 6aa5441c-613e-4a1f-9d94-5d04d5a8cc3a
W = ot.OTDD._getW(mnist_x,mnist_y, fmnist_x, fmnist_y);

# ╔═╡ 9098a67f-de1f-4c06-b8ab-266ff0372231
C, γ, otdd_initial = ot.OTDD.otdd(mnist_x,mnist_y, fmnist_x, fmnist_y, W=W);

# ╔═╡ e7ab7253-70b7-4ad8-8c49-4910a2aa68d0
round(otdd_initial,digits=2)

# ╔═╡ f1cf5f97-dd45-4d0d-beea-3640ff5aa96e


# ╔═╡ ad4fe979-f23e-4a68-a69b-b98d3406c90b


# ╔═╡ Cell order:
# ╟─2ffddf10-bd51-11eb-12cb-f1add38b47fb
# ╟─b3a49e8b-b54c-4247-8370-c2a917e57056
# ╠═8529c382-f72f-44f7-8ccd-ce68ab03776e
# ╠═1fbf4fa5-3eb5-4866-9b55-4bf9c5967250
# ╠═e5494bbe-ce7d-4a63-8b9f-c0989b3acffb
# ╠═6aa5441c-613e-4a1f-9d94-5d04d5a8cc3a
# ╠═9098a67f-de1f-4c06-b8ab-266ff0372231
# ╠═e7ab7253-70b7-4ad8-8c49-4910a2aa68d0
# ╟─b3fd7749-18ef-4033-9e2d-431ee284c11b
# ╟─4a741e16-7e80-43cb-bfe3-63ae442d61f1
# ╟─75b36234-7d5b-4527-9274-2239046b556a
# ╠═068369ca-a6db-4f01-b192-1256332202f0
# ╟─b89edb81-2e62-4b2a-8ce3-3e4c25a31b55
# ╠═df0f24fd-f847-40fb-b3dc-12350face55f
# ╠═c315a543-e9be-4b79-8449-b9175c923bb8
# ╟─29ac65a4-a1c1-47a8-a691-be90f988709f
# ╟─21b3b741-1ea1-49a4-a6ae-b22666f53e19
# ╟─8feb1c33-ba6d-449a-8676-b1144d4d4312
# ╟─3ce0657e-5487-43c9-a28c-7661c95a1486
# ╟─c1c693c0-1c57-43c8-af20-9cd5e9c7d6af
# ╟─742ef2ec-4c23-46e7-ad39-ff838ef156b1
# ╟─7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
# ╟─a9cb0024-ae23-4fc9-81d8-4ea335884900
# ╠═95063639-9e69-4bff-85e0-31e642be8a0a
# ╠═839f0087-5890-462d-8507-70b3c3db797d
# ╠═07120a08-226b-4907-87c7-f5d63af616a7
# ╠═8a0319c9-51ac-4e99-a4d9-1c07bee83381
# ╠═7b20f2d5-a2c7-4705-9576-f39cf4ca03f5
# ╠═3aed32f1-f15c-4672-8641-e52c9a7c7671
# ╠═b2aafd47-c5c1-4ada-8d48-bfea30292d20
# ╠═bf62c705-49cc-4545-8bdf-316a61c9a5c0
# ╠═f6259df8-8fee-48d0-b5ec-04e0dbde1250
# ╠═1150fee9-a9a5-4158-be90-eb72385cf3d1
# ╠═589be23b-fc9b-4c5b-9316-64ce3074a281
# ╠═d468ee56-e522-421b-91b3-66135b0e8683
# ╠═835d761d-bfe5-45f6-919d-d0c03711a5c8
# ╠═70a5b623-418e-4b91-a1b2-dd88a26d5756
# ╠═57103a08-fefc-4c8c-85cb-a054de9edc5b
# ╠═583923b6-f08a-4074-94ff-2fc480e16277
# ╠═e526398c-e77e-43fe-bfb4-638ff2ad579b
# ╠═6c333ac9-22a6-4353-a4f1-67bebdaadc24
# ╠═1b066cf7-bf1b-4445-9e58-275679838973
# ╠═f1cf5f97-dd45-4d0d-beea-3640ff5aa96e
# ╠═ad4fe979-f23e-4a68-a69b-b98d3406c90b
