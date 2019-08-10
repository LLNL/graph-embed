
# Description

graph-embed is a small library for (multi-level) graph partitioning and embedding.

## Dependencies

[linalgcpp](github.com/gelever/linalgcpp)
[OpenMP](www.openmp.org/)
[Plotly](plot.ly/)

## Installation

To build the project:
- mkdir build && cd build
- cmake ../
- make -j9

## graph-embed

Graphs can be read in using linalgcpp.
Once imported, partition the graph using `partition::partition` which takes a graph and a coarsening factor. This returns a hierarchy of partitions of the given graph, which each coarsen approximately by the given coarsening factor.
These can be used to create a hierarchy of graphs.
Finally, call `partition::embed` with the hierarchy of graphs, hierarchy of partitions, and desired dimension. This returns a vector of coordinates, indexed by the vertices of the graph.

See examples/embed.cpp and examples/embedder.cpp for an example.

## Authors

graph-embed was created by Benjamin Quiring (quiring1@llnl.gov) at LLNL.

## Getting Involved

Please email quiring1@llnl.gov if you are interested in this project.

## Contributing

Feel free to open a pull request to contribute.

## License
This project is licensed under the LGPL-2.1 License. See the LICENSE file for details.

## Release
LLNL-CODE-781679




