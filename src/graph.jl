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

    edge_feature_keys = filter(k -> startswith(k, "edge|"), keys(data)) # get edge_feature_keys (those starting with edge|)

    if length(edge_feature_keys) > 0
        if length(edge_feature_keys) > 1
            error("More than one edge key found: $(edge_feature_keys). Not yet implemented for more than one.")
        elseif length(edge_feature_keys) == 1
            senders, receivers = parse_custom_edges_features(data)
            edge_feature_key = first(edge_feature_keys)
            edge_features = data[edge_feature_key]  # Todo: sollte doppelt gemoppelt sein. Überprüfen
        else
            println("Something went wrong with number of edges?")
        end

    elseif haskey(data, "cells")
        senders, receivers = triangles_to_edges(data["cells"][:, :, 1])
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end
        rel_vec = [data["mesh_pos"][:, senders[i], 1] -
                   data["mesh_pos"][:, receivers[i], 1] for i in eachindex(senders)]
    elseif haskey(data, "edges")
        senders, receivers = parse_edges(data["edges"])
        if 0 in senders || 0 in receivers
            senders .+= 1
            receivers .+= 1
        end
        rel_vec = [data["mesh_pos"][:, senders[i], 1] -
                   data["mesh_pos"][:, receivers[i], 1] for i in eachindex(senders)]
        relative_mesh_pos = hcat(rel_vec...)
        edge_features = vcat(
            relative_mesh_pos, permutedims(map(norm, eachcol(relative_mesh_pos))))
        println("size ef: ", size(edge_features))
        println("tpye ", typeof(edge_features))

    else
        throw(KeyError("Data does not contain cell or edge information!"))
    end

    data["node_type"] = device(node_type)
    data["senders"] = device(senders)
    data["receivers"] = device(receivers)
    data["edge_features"] = device(edge_features)
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
function build_graph(mgn::GraphNetwork, data, fields, datapoint::Integer, node_type,
        edge_features, senders::AbstractArray{T, 1},
        receivers::AbstractArray{T, 1}) where {T <: Integer}
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
    for field in fields
        # nf = vcat(nf, (data[field][1:datapoint, :]))  # Should work for solvertraining, since given data has dict{String, 5x26} format
        nf = vcat(nf, (data[field][(datapoint - 4):datapoint, :]))  # Should work for Derivative
        # nf = vcat(nf, (data[field][:, :, min(size(data[field], 3), datapoint)]))  # Original Line of code, but dont think it works
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
        edge_features,
        senders,
        receivers
    )
end
