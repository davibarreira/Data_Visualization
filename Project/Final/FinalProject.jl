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

# ╔═╡ 2ffddf10-bd51-11eb-12cb-f1add38b47fb
md"""
# Dataset Transferability Analysis
A visual tool for improving transfer learning via data augumentation and Optimal Transport
"""

# ╔═╡ b3a49e8b-b54c-4247-8370-c2a917e57056
md"""
### Installing and Importing Packages and Parsing Data
"""

# ╔═╡ 3770fcc8-a00a-4b9f-9dad-687916e0257a
@bind markpicker Select(["Images","Circles"])

# ╔═╡ f2fdc529-f0ac-4860-9cbc-4cb2e98abaf9
md"""
#### Finals Picks to be Augumented:
After pressing the "Final Picks" button, the user choses which images he wants to augument.
"""

# ╔═╡ 4adf778c-bf01-4b93-93c6-1c6cd326e184
md"""
#### Aplying Augumentation to Final Selection
"""

# ╔═╡ ba9df336-c132-4ff0-9f34-6b8b5b0b34a7
@bind default Button("Default")

# ╔═╡ d58935cf-d11a-4262-9389-e2f17b33d800
md"""
Press the buttom above to restore the $(default).

`Gamma Transform` = $(@bind gamma Slider(0.1:0.1:15, default=1))

`Rotation Angle ` = $(@bind rotation Slider(0:pi/10:2*pi, default=0))

`Mirror Vertical` = $(@bind mirrorv html"<input type=checkbox >") ;
`Mirror Horizontal` = $(@bind mirrorh html"<input type=checkbox >")
"""

# ╔═╡ bf62c705-49cc-4545-8bdf-316a61c9a5c0
@bind savetransformation Button("Save Modifications")

# ╔═╡ caa1aad4-7c09-41ce-8b59-2ec257fefc87
md"""
## Dataset - Import and Wrangling
"""

# ╔═╡ 835d761d-bfe5-45f6-919d-d0c03711a5c8
md"""
### Auxiliary Functions
"""

# ╔═╡ 4ca60e23-30dd-427f-ac5c-ea68c8f3e2b7
function applygammatransform(image, gamma = 2)
    mimg = adjust_histogram(image, GammaCorrection(gamma = gamma))
    return mimg
end

# ╔═╡ d0267b4a-8e52-42d7-9a36-7ca3259f3f31
function applyrotate(image, rotation=pi/2)
    trfm = recenter(RotMatrix(rotation), center(image));
    mimg = warp(image,trfm)
    mimg = mimg[1:size(image)[1],1:size(image)[2]]
    return mimg
end

# ╔═╡ f3c8e3ad-d890-43eb-881e-56f89b8a0f98
function applytransformations(image, gamma = 0.1, rotation = 0, mirrorv=false, mirrorh=false)
	mimg = applyrotate(image,rotation)
    mimg = applygammatransform(mimg,gamma)
	if mirrorv == true
		mimg = reverse(mimg,dims=1)
	end
	if mirrorh == true
		mimg = reverse(mimg,dims=2)
	end
    return mimg
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

# ╔═╡ 8529c382-f72f-44f7-8ccd-ce68ab03776e
begin
    import Pkg
    Pkg.activate(".")
    using MLDatasets, VegaLite, DataFrames, Distances, LinearAlgebra, PlutoUI, HypertextLiteral, JSON, JSONTables, Images, OptimalTransport, UMAP, ImageContrastAdjustment, ImageCore, ImageTransformations, Rotations, CoordinateTransformations
	ot = ingredients("./otdd.jl");
end

# ╔═╡ e5494bbe-ce7d-4a63-8b9f-c0989b3acffb
begin
	mnist_x = reshape(MNIST.traintensor(Float64),28*28,:);
	mnist_y = MNIST.trainlabels(1:size(mnist_x, 2));
	fmnist_x = reshape(FashionMNIST.traintensor(Float64),28*28,:);
	fmnist_y = FashionMNIST.trainlabels(1:size(fmnist_x, 2));

	N = 200;
	mnist_x  = mnist_x'[1:N,:];
	mnist_y  = mnist_y[1:N];
	fmnist_x = fmnist_x'[1:N,:];
	fmnist_y = fmnist_y[1:N];
	img_url = vcat(
					["http://localhost:5000/mnist_"*string(i)*".png" for i in 1:N],
					["http://localhost:5000/fmnist_"*string(i)*".png" for i in 1:N]);
	W = ot.OTDD._getW(mnist_x,mnist_y, fmnist_x, fmnist_y);
	C, γ, otdd_initial = ot.OTDD.otdd(mnist_x,mnist_y, fmnist_x, fmnist_y, W=W);
	res = umap(hcat(mnist_x',fmnist_x'); n_neighbors=10, min_dist=0.001, n_epochs=200)';
	
	GetSelected(text="Select Initial Samples") = @htl("""
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
	
	GetFinalSelection(text="Final Picks") = @htl("""
	<div id="ok">
	<button>$(text)</button>
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
    <script id="selection">
	var div = currentScript.parentElement
	var button = div.querySelector("button")
	button.addEventListener("click", (e) => {
		div.value = d3.select("#sampleview").selectAll("svg").selectAll(".selected").data()
		div.dispatchEvent(new CustomEvent("input"))
		e.preventDefault()
	})
	const svg = d3.select("#sampleview").selectAll("svg").selectAll(".selected")
	div.value = svg.data()
    </script>
	</div>
""")
	
end;

# ╔═╡ dd32009d-8f8e-4745-bc8d-3bfa1ce82ee1
md"""
## Intial OTDD = $(round(otdd_initial,digits=2))
Remember, the OTDD measures the distance between datasets, so trying to minimize is a heuristic to improve the process of transfer learning.
Hence, the goal of this project is to create a visualization tool to help analystis understand their dataset and perform data augumentation, seeking to reduce the OTDD and which (hopefully) can lead to better transfer learning.
"""

# ╔═╡ a9cb0024-ae23-4fc9-81d8-4ea335884900
@bind selected GetSelected()

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
	
		var height = 200;
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
	    
      value = myimage.attr('opacity',0.1)
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

# ╔═╡ e314152b-9f97-43cd-a164-8833d13c1eb0
@bind finalselection GetFinalSelection()

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

# ╔═╡ 5601474e-ac92-4356-8e63-d91ca3111fa8
begin
	aug_df = Dict(:dataset=>[],:array=>[], :img=>[], :id=>[], :label=>[])
	datasets_x = Dict("mnist"=> mnist_x, "fmnist"=>fmnist_x)
	if length(finalselection)>0
		for i in 1:length(finalselection)
			_dataset = finalselection[i]["dataset"]
			imgid = finalselection[i]["id"]
			if _dataset == "mnist"
				img = MNIST.convert2image(datasets_x[_dataset][imgid,:])
			else
				img = MNIST.convert2image(datasets_x[_dataset][imgid-N,:])
			end
			mimg = applytransformations(img, gamma, rotation, mirrorv, mirrorh)
			aimg = replace!(reshape(convert(Array{Float64}, mimg)',28*28),NaN=>0)
			push!(aug_df[:dataset],_dataset)
			push!(aug_df[:array],aimg)
			push!(aug_df[:img],mimg)
			push!(aug_df[:id],imgid)
			push!(aug_df[:label],finalselection[i]["label"])
		end
	end
	aug_df = DataFrame(aug_df)
end;

# ╔═╡ 82f629f5-204a-433d-860f-4773326de455
aug_df[:,:img]

# ╔═╡ 4c11b8f3-e8e6-4932-acb5-b6de350efe8c
begin
	savetransformation 
	augmnist_x = copy(mnist_x)
	augfmnist_x = copy(fmnist_x)
	_idmnist = aug_df[aug_df[:,:dataset] .== "mnist",:id]
	_idfmnist = aug_df[aug_df[:,:dataset] .== "fmnist",:id] .- N
	
	if length(_idmnist) > 1
		augmnist_x[_idmnist,:] .= hcat(aug_df[aug_df[:,:dataset] .== "mnist",:array]...)'
	end
	if length(_idfmnist) > 1
		augmnist_x[_idfmnist,:] .= hcat(aug_df[aug_df[:,:dataset] .== "fmnist",:array]...)'
	end
	Cfinal, γfinal, otddfinal = ot.OTDD.otdd(augmnist_x,mnist_y, augfmnist_x, fmnist_y, W=W);
end;

# ╔═╡ 7a410198-26f4-4319-9739-851a87359c67
md"""
# Did the modifications improve the results?

#### Final OTDD = $(round(otddfinal,digits=2))
#### Intial OTDD = $(round(otdd_initial,digits=2))
"""

# ╔═╡ 5fce3b27-cef6-46e2-8776-9e185902e414
let
	savetransformation
	if size(aug_df)[1] > 0
		for row in eachrow(aug_df)
			save("./images/modified/mnist_"*string(row[:id])*".png",
				row[:img])
		end
	end
end

# ╔═╡ f1cf5f97-dd45-4d0d-beea-3640ff5aa96e
"""
    CreateEdges(μ,ν,γ)
Creates the edges for plotting.
μ and ν correspond to the positions of the mass
of the distributions.
"""
function CreateEdges(μ, ν, γ; ewfilter=0)
    edges = Array{Float64}(undef, 0, 2)
    pe    = []
    source    = []
    target    = []
    for i in 1:size(μ)[1], j in 1:size(ν)[1]
        edges  = vcat(edges,[μ[i,1],μ[i,2]]')
        edges  = vcat(edges,[ν[j,1],ν[j,2]]')
        pe     = vcat(pe,string([i,j]))
        pe     = vcat(pe,string([i,j]))
        source = vcat(source,i)
        source = vcat(source,i)
        target = vcat(target,j)
        target = vcat(target,j)
    end
    df = DataFrame(edges_x=edges[:,1],edges_y = edges[:,2],pe=pe, source=source, target=target);
    edge_w = []
    for i in 1:size(γ)[1], j in 1:size(γ)[1]
        edge_w = vcat(edge_w,γ[i,j])
        edge_w = vcat(edge_w,γ[i,j])
    end
    df[!,"ew"] = edge_w./maximum(edge_w);
    
    filter = ewfilter
    df = df[df[:,:ew] .>= filter,:];
    df[!,:id] .= 1:size(df)[1];
    return df
end


# ╔═╡ ad4fe979-f23e-4a68-a69b-b98d3406c90b
function getlargerew(edges, n=1)
	return combine(groupby(edges, :source)) do sdf
		first(sort(sdf,:ew; rev=true),n)
	end
end

# ╔═╡ 564c6127-b38b-4773-baa8-75e7a17dd677
begin
	d_mnist  = Matrix(df[df[:,:dataset].=="mnist",[:x,:y]])
	d_fmnist = Matrix(df[df[:,:dataset].=="fmnist",[:x,:y]]);
	edges = getlargerew(CreateEdges(d_mnist, d_fmnist, γ, ewfilter=0.1),2);
end;

# ╔═╡ c5f10b93-4e80-49a5-9f95-fbc489449bde
edjson = arraytable(edges)

# ╔═╡ 239deeeb-b34f-4057-9752-4d6f5e0b916d
begin
	df[!,:pe] = edges[!,:pe]
	df[!,:source] = edges[!,:source]
	df[!,:target] = edges[!,:target]
	tx = []
	ty = []
	tl = []
	Ndf  = Int(size(df)[1]/2)
	
	
	f(x) = argmax(γ[x,:])
	g(x) = argmax(γ[:,x])
	mnistorigin = collect(1:N)
	fmnistorigin = collect(1:N)
	mnistfinal = f.(mnistorigin);
	fmnistfinal = g.(fmnistorigin);
	df[!,:origin] = vcat(mnistorigin,fmnistorigin)
	df[!,:final] = vcat(mnistfinal,fmnistfinal);
	for i in 1:size(df)[1]
		if df[i,:dataset] == "mnist"
			push!(tx,df[df[i,:final]+Ndf,:x])
			push!(ty,df[df[i,:final]+100,:y])
			push!(tl,df[df[i,:final]+Ndf,:label])
		else
			push!(tx,df[df[i,:origin],:x])
			push!(ty,df[df[i,:origin],:y])
			push!(tl,df[df[i,:origin],:label])
		end
	end
	df[!,:tx] = tx
	df[!,:ty] = ty
	df[!,:tl] = tl
	
	source = df[1:N,:];
	source[!,:fmnist_label] = df[source[!,:final].+N,:label];
	source[!,:px] = source[:,:label] + rand(N)*0.8
	source[!,:py] = source[:,:fmnist_label] + rand(N)*0.8;
end;

# ╔═╡ b3e7ad58-ac03-463c-9df9-bdf5872a23ed
c1 = @vlplot("data"=source,"height"=350,"width"=350,"background"="white",
    "mark"={:rect},
    "x"={"field"=:label,"type"="ordinal","sort"="ascending",
		"axis"={"orient"="top","labelAngle"=0}},
    "y"={"field"=:fmnist_label,"type"="ordinal","sort"="ascending"},
    "color"={"field"=:label, aggregate="count", "scale"={scheme="lightgreyteal"}},
    "config"= {"axis"= {"grid"= true, "tickBand"= "extent"}}
);

# ╔═╡ 2b880632-e9d0-40f8-8231-315ad2abc6b0
nedges =[Dict("values"=> [Dict("ex"=>edges[i,:edges_x],"ey"=>edges[i,:edges_y]),
			Dict("ex"=>edges[i+1,:edges_x],"ey"=>edges[i+1,:edges_y])])
			for i in 1:2:size(edges)[1]]

# ╔═╡ 7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
Scatter = @htl("""
	
<style type="text/css">
#wrap {
   width:900px;
   margin:0 auto;
}
#left_col {
   float:left;
   width:500px;
}
#right_col {
   float:right;
   width:400px;
}
.dotfmnist {
  height: 10px;
  width: 10px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
}
.dotmnist {
  height: 10px;
  width: 10px;
  background-color: #4682b4;
  border-radius: 50%;
  display: inline-block;
}
</style>

<div id="wrap">
	<h1> Transferability Analysis via Optimal Transport </h1>
    <div id="left_col">
	<h5>Optimal Transport between MNIST and FMNIST</h5>
	<p>
	mnist <span class="dotmnist"></span> 	fmnist <span class="dotfmnist"></span>
     <div id="myvis"></div>
    </div>
    <div id="right_col">
	<h5>Optimal Transport coupling matrix heatmap</h5>
	<div id="myvis2">
	</div>
    </div>
</div>

<div>
    <script src="https://cdn.jsdelivr.net/npm/vega@5.20.2"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.1.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.17.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>
    <script>
        let cell = currentScript.closest("pluto-cell");
        cell.style.width = "1000px";
    </script>

        <script id="createplot">
            var div = currentScript.parentElement;

            var selection = 0;

            var height = 400;
            var width = 350;
            var margin = { top: 20, right: 30, bottom: 30, left: 40 };

            const data = JSON.parse($(dfjson));
            const edges = JSON.parse($(json(nedges)));

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
                .attr("stroke", "white");

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

            const color = d3.scaleOrdinal().domain(["mnist", "fmnist"]).range(["steelblue", "grey"]);

            var line = d3
                .line()
                .x((d) => x(d.ex))
                .y((d) => y(d.ey));

            const path = svg
                .append("g")
                .selectAll("path")
                .data(edges)
                .enter()
                .append("path")
                .attr("d", (d) => line(d.values));

            path.attr("stroke", "steelblue").attr("stroke-width", 0.5).attr("stroke-linejoin", "round").attr("stroke-linecap", "round");

		if($(markpicker) == "Circles"){
	
	var myimage = svg
                .append("g")
                .selectAll("circle")
                .data(data)
                .join("circle")
                .attr("cx", (d) => x(d.x))
                .attr("cy", (d) => y(d.y))
                .attr("r", 10)
				.attr("fill", "steelblue")
                .attr("opacity", 0.8)
				.style("fill", function (d) { return color(d.dataset) } )
                .attr("id", function (d, i) {
                    return "soruce" + d.source;
                });	
	
	
	}else{
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
                .attr("opacity", 1)
                .attr("id", function (d, i) {
                    return "soruce" + d.source;
                });}

            function brushed({ selection }) {
                let value = [];
                if (selection) {
                    const [[x0, y0], [x1, y1]] = selection;

                    value = myimage
                        .attr("opacity", 0.1)
                        .attr("class", "unselected")
                        .filter(function (d) {
                            return x0 <= x(d.x) && x(d.x) < x1 && y0 <= y(d.y) && y(d.y) < y1;
                        })
                        .attr("class", "selected")
                        .attr("opacity", 1.0)
                        .data();
                } else {
                    myimage.attr("class", "selected").attr("opacity", 1.0);
                }
                div.value = value;
                svg.property("value", value).dispatch("input");
            }
            const brush = d3.brush().on("start brush end", brushed);

            svg.call(brush);

            div.value = svg.selectAll("selected");
        </script>
        <script type="text/javascript">
            const spec = JSON.parse($(json(c1)));
            vegaEmbed("#myvis2", spec)
                .then((result) => console.log(result))
                .catch(console.warn);
        </script>
    </div>
</div>

""")

# ╔═╡ Cell order:
# ╟─2ffddf10-bd51-11eb-12cb-f1add38b47fb
# ╟─b3a49e8b-b54c-4247-8370-c2a917e57056
# ╠═8529c382-f72f-44f7-8ccd-ce68ab03776e
# ╠═c315a543-e9be-4b79-8449-b9175c923bb8
# ╠═c5f10b93-4e80-49a5-9f95-fbc489449bde
# ╟─dd32009d-8f8e-4745-bc8d-3bfa1ce82ee1
# ╟─3770fcc8-a00a-4b9f-9dad-687916e0257a
# ╟─7a1129a6-e48a-4d1c-8d8e-d9c656a47dee
# ╟─a9cb0024-ae23-4fc9-81d8-4ea335884900
# ╟─e314152b-9f97-43cd-a164-8833d13c1eb0
# ╟─95063639-9e69-4bff-85e0-31e642be8a0a
# ╟─b3e7ad58-ac03-463c-9df9-bdf5872a23ed
# ╟─f2fdc529-f0ac-4860-9cbc-4cb2e98abaf9
# ╟─4adf778c-bf01-4b93-93c6-1c6cd326e184
# ╟─ba9df336-c132-4ff0-9f34-6b8b5b0b34a7
# ╟─d58935cf-d11a-4262-9389-e2f17b33d800
# ╟─82f629f5-204a-433d-860f-4773326de455
# ╟─bf62c705-49cc-4545-8bdf-316a61c9a5c0
# ╟─7a410198-26f4-4319-9739-851a87359c67
# ╟─4c11b8f3-e8e6-4932-acb5-b6de350efe8c
# ╟─caa1aad4-7c09-41ce-8b59-2ec257fefc87
# ╠═e5494bbe-ce7d-4a63-8b9f-c0989b3acffb
# ╠═df0f24fd-f847-40fb-b3dc-12350face55f
# ╠═239deeeb-b34f-4057-9752-4d6f5e0b916d
# ╟─835d761d-bfe5-45f6-919d-d0c03711a5c8
# ╠═d468ee56-e522-421b-91b3-66135b0e8683
# ╠═4ca60e23-30dd-427f-ac5c-ea68c8f3e2b7
# ╠═f3c8e3ad-d890-43eb-881e-56f89b8a0f98
# ╠═d0267b4a-8e52-42d7-9a36-7ca3259f3f31
# ╠═5601474e-ac92-4356-8e63-d91ca3111fa8
# ╠═5fce3b27-cef6-46e2-8776-9e185902e414
# ╠═1b066cf7-bf1b-4445-9e58-275679838973
# ╠═f1cf5f97-dd45-4d0d-beea-3640ff5aa96e
# ╠═ad4fe979-f23e-4a68-a69b-b98d3406c90b
# ╠═564c6127-b38b-4773-baa8-75e7a17dd677
# ╠═2b880632-e9d0-40f8-8231-315ad2abc6b0
