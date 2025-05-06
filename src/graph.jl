#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Statistics: norm

"""
    create_base_graph(data, type_size, type_min, device)

Constructs the parts of the node features and edge features that do not change during one trajectory.

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `type_size`: Depth of the node type matrix.
- `type_min`: Offset of the node type matrix.
- `device`: Device where the normaliser should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).

## Returns
- Onehot vector of the node types used for node features.
- Vector of indices where each edge in the graph starts.
- Vector of indices where each edge in the graph ends.
- Array of edge features for each edge in the graph.
"""
function create_base_graph!(data, type_size, type_min, device::Function)
    node_type = one_hot(
        vec(data["node_type"][:, :, 1]), type_size - type_min + 1, 1 - type_min)

    if haskey(data, "cells")
        senders, receivers = triangles_to_edges(data["cells"][:, :, 1])
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end
        rel_vec = [data["mesh_pos"][:, senders[i], 1] -
                   data["mesh_pos"][:, receivers[i], 1] for i in eachindex(senders)]
        relative_mesh_pos = hcat(rel_vec...)

        mesh_features = vcat(
            relative_mesh_pos, permutedims(map(norm, eachcol(relative_mesh_pos))))
    elseif haskey(data, "edges")
        senders, receivers = parse_edges(data["edges"])
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end
        rel_vec = [data["mesh_pos"][:, senders[i], 1] -
                   data["mesh_pos"][:, receivers[i], 1] for i in eachindex(senders)]
        relative_mesh_pos = hcat(rel_vec...)
        mesh_features = vcat(
            relative_mesh_pos, permutedims(map(norm, eachcol(relative_mesh_pos))))
    else
        throw(KeyError("Data does not contain cell or edge information!"))
    end

    data["node_type"] = device(node_type)
    data["senders"] = device(senders)
    data["receivers"] = device(receivers)
    data["mesh_features"] = device(mesh_features)
    # return device(node_type), device(senders), device(receivers), device(edge_features)
end

"""
    build_graph(mgn, data, fields, datapoint, node_type, edge_features, senders, receivers)

Constructs a [FeatureGraph](@ref) based on the given arguments.

## Arguments
- `mgn`: MGN from where normalisers for node & edge features are used.
# - `data`: Data from the dataset containing one trajectory.
- `data`: Dict containing the features from the last x steps. e.g. data["energy_output"][1:5, 1:26]
- `fields`: Node features of the MGN.
- `datapoint`: Current index of the data corresponding to the current timestep.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph. (mesh_pos expects: edge_features::AbstractArray{Float32, 2})
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.

## Returns
- Resulting [FeatureGraph](@ref).
"""
function build_graph_old(
        mgn::GraphNetwork, data, fields, datapoint::Integer, node_type, ef,
        senders::AbstractArray{T, 1}, receivers::AbstractArray{T, 1}) where {T <: Integer}
    # Removed generator in favor of removing Zygote.jl piracies (minimal increase of time and allocations)
    # Can be reverted once Enzyme.jl is compatible
    #nt = mgn.n_norm["node_type"](node_type)

    nt = node_type
    nf = similar(nt, 0, size(nt, 2))
    # for field in fields
    #     nf = vcat(
    #         nf, mgn.n_norm[field](data[field][:, :, min(size(data[field], 3), datapoint)]))
    # end

    # Notizen: Solvertraining: datapoint ist 5. data[field] ist 5x26. neuester state ist an idx 5. Füge in nf den letzten state ein. NF ist auch 5x26

    # if idx > 1
    #     idx_iter = idx-4:idx
    sleep(4)
    for field in fields
        # nf = vcat(nf, (data[field][1:datapoint, :]))  # Should work for solvertraining, since given data has dict{String, 5x26} format
        # nf = vcat(nf, (data[field][(datapoint - 4):datapoint, :]))  # Should work for Derivative
        nf = vcat(nf, (data[field][:, :, min(size(data[field], 3), datapoint)]))  # Original Line of code
    end
    # println("size nf: ", size(nf))
    # println("type data: ", typeof(data))
    # println("size data[energy_output]: ", size(data["energy_output"]))
    # println("energy node 1 of 1:5: ", (data["energy_output"][1:datapoint, 1]))
    # sleep(4)

    # Todo: Hardcoded; Reshape needed to get it into same format as nt, as in (feature x no_nodes)
    # nf_reshaped = permutedims(nf, (1, 3, 2))
    # nf_flat = reshape(nf_reshaped, :, size(nf, 2))
    # nf_result = nf_flat  # (features * timesteps, nodes)
    nf = vcat(nf, nt)

    # println("end of build_graph")
    return FeatureGraph(
        nf,
        # vcat(
        #     [mgn.n_norm[field](data[field][:, :, min(size(data[field], 3), datapoint)]) for field in fields]...,
        #     mgn.n_norm["node_type"](node_type)
        # ),
        ef,
        senders,
        receivers
    )
end

function build_graph(mgn::GraphNetwork, data, fields, datapoint::Integer,
        node_type, edge_features, senders::AbstractArray{T, 1},
        receivers::AbstractArray{T, 1}) where {T <: Integer}
    nt = mgn.n_norm["node_type"](node_type)
    # nf = CUDA.zeros(Tnf, 0, size(nt, 2))
    nf = similar(nt, 0, size(nt, 2))

    for field in fields
        @assert datapoint≤size(data[field], 3) "Datapoint $datapoint out of bounds for field $field"
        nf = vcat(
            nf, mgn.n_norm[field](data[field][:, :, min(size(data[field], 3), datapoint)]))
    end
    nf = vcat(nf, nt)

    ef = mgn.e_norm["mesh_pos"](edge_features)  # Todo: Sollte doch zwischen -1 und 1 sein? ist aber auch bei 1.7? legal?

    # edge_features = convert(typeof(nf), edge_features)

    # Todo: Edge_Features anpassen, Normierung ähnlich wie bei node_features

    return FeatureGraph(
        nf,
        ef,
        senders,
        receivers
    )
end

# function build_graph(mgn::GraphNetwork, data, fields, datapoint::Integer, node_type, ef,
#         senders::AbstractVector{T}, receivers::AbstractVector{T}) where {T <: Integer}
#     println("Fields in build_graph: ", fields)
#     println("Keys of data: ", keys(data))
#     sleep(2)

#     # Starte mit leerem Node-Feature-Array
#     nf = similar(node_type, 0, size(node_type, 2))

#     # Füge alle spezifizierten Features hinzu
#     for field in fields
#         nf = vcat(nf, data[field][:, :, min(datapoint, size(data[field], 3))])
#     end

#     # Hänge node_type als Features hinten dran
#     nf = vcat(nf, node_type)

#     println("Size of nf: ", size(nf))

#     return FeatureGraph(
#         nf,
#         ef,
#         senders,
#         receivers
#     )
# end
