### A Pluto.jl notebook ###
# v0.14.5

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

# ╔═╡ 0546ddd0-bbf8-11eb-2090-bb8f5821c377
using JSON, PlutoUI, HypertextLiteral

# ╔═╡ 9fc82a2e-90b7-4458-95fe-bbc3a759e5a8
@bind fantastic_x Slider(0:400)

# ╔═╡ 803b6800-e4d5-449c-8b7c-3bd721967e56
my_data = [
	(name="Cool", coordinate=[100, 100]),
	(name="Awesome", coordinate=[200, 100]),
	(name="Fantastic!", coordinate=[fantastic_x, 150]),
]

# ╔═╡ caaa645f-ccd2-41d1-aa31-8fed09aa8d11
JSON.json(my_data)

# ╔═╡ 9ee39a46-5788-47ec-8ae0-d8385f782b58
@htl("""
	<script src="https://cdn.jsdelivr.net/npm/d3@6.2.0/dist/d3.min.js"></script>

	<script>

	// interpolate the data 🐸
	const data = JSON.parse($(JSON.json(my_data)))

	const svg = DOM.svg(600,200)
	const s = d3.select(svg)

	s.selectAll("text")
		.data(data)
		.join("text")
		.attr("x", d => d.coordinate[0])
		.attr("y", d => d.coordinate[1])
		.text(d => d.name)

	return svg
	</script>
""")

# ╔═╡ bdc54a3b-68ea-4737-90ef-516029a4b637


# ╔═╡ Cell order:
# ╠═0546ddd0-bbf8-11eb-2090-bb8f5821c377
# ╠═9fc82a2e-90b7-4458-95fe-bbc3a759e5a8
# ╠═caaa645f-ccd2-41d1-aa31-8fed09aa8d11
# ╠═803b6800-e4d5-449c-8b7c-3bd721967e56
# ╠═9ee39a46-5788-47ec-8ae0-d8385f782b58
# ╠═bdc54a3b-68ea-4737-90ef-516029a4b637
