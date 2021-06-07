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

        var height = 500;
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
            .attr("width", 30)
            .attr("height", 30)
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
                    .attr("opacity", 0.5)
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

        //svg.call(d3.zoom().extent([[0, 0], [width, height]]).translateExtent([[0, 0], [width, height]])
        //.scaleExtent([1, 8]).on("zoom", zoomed)).on("touchstart.zoom", null).on("mousedown.zoom", null)
        //.on("dblclick.zoom", null);

        function zoomed({ transform }) {
            myimage.attr("transform", transform);
        }
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
            .attr("width", 30)
            .attr("height", 30)
            .attr("opacity", 1);

        div.value = svg.selectAll("selected");
    </script>
</div>
