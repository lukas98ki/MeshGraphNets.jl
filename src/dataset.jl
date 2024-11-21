#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Distributions: Normal
import HDF5: Group
import Random: MersenneTwister
import TFRecord: Example

import HDF5: read_dataset
import JLD2: jldopen
import JSON: parse
import Random: seed!, make_seed, shuffle
import TFRecord: read

include("strategies.jl")

"""
    Dataset(file, file_valid, meta, ch, ch_valid, data, data_valid, cs, current, current_valid)

Data structure for the training, evaluation and test data inside a dataset.

## Arguments
- `file`: Path of training or test data file (depending on the function call to [load_dataset](@ref)).
- `file_valid`: Path of validation data file.
- `meta`: Metadata of the dataset.
- `ch`: Channel that reads trajectories from the data file.
- `ch_valid`: Channel that reads trajectories from the validation data file.
- `data`: Dictionary that stores trajectories that were already read from the data file.
- `data_valid`: Dictionary that stores trajectories that were already read from the validation data file.
- `cs`: Size of the data channels.
- `current`: Index of current trajectory.
- `current_valid`: Index of current validation trajectory.
"""
struct Dataset
    meta::Dict{String, Any}
    datafile::String
    lock::ReentrantLock
end

function Dataset(datafile::String, metafile::String, args)
    if !isfile(datafile)
        throw(ArgumentError("Invalid datafile: $datafile"))
    elseif !endswith(datafile, ".jld2") || !endswith(datafile, ".h5")
        throw(ArgumentError("Invalid file format for datafile: $datafile. Possible formats are [.jld2, .h5]"))
    end
    if !isfile(metafile)
        throw(ArgumentError("Invalid metafile: $metafile"))
    elseif !endswith(metafile, ".json")
        throw(ArgumentError("Invalid file format for metafile: $metafile. Possible formats are [.json]"))
    end

    meta = parse(Base.read(metafile), String)
    keys_traj = keystraj(datafile)
    meta["n_trajectories"] = length(keys_traj)
    meta["keys_trajectories"] = keys_traj
    merge!(meta, Dict(String(key) => getfield(args, key) for key in propertynames(args)))

    Dataset(meta, datafile, ReentrantLock())
end

function Dataset(split::Symbol, path::String, args)
    if split != :train && split != :valid && split != :test
        throw(ArgumentError("Invalid symbol for dataset: $split. Possible values are [:train, :valid, :test]"))
    end
    if !isfile(joinpath(path, "meta.json"))
        throw(ArgumentError("Metafile not found in path: $path. Check that your metafile is named \"meta.json\""))
    end

    meta = parse(Base.read(joinpath(path, "meta.json"), String))
    datafile = get_file(split, path)
    keys_traj = keystraj(datafile)
    meta["n_trajectories"] = length(keys_traj)
    meta["keys_trajectories"] = keys_traj
    merge!(meta, Dict(String(key) => getfield(args, key) for key in propertynames(args)))

    Dataset(meta, datafile, ReentrantLock())
end

function get_file(split::Symbol, path::String)
    filename = String(split)
    if isfile(joinpath(path, "$filename.jld2"))
        return joinpath(path, "$filename.jld2")
    elseif isfile(joinpath(path, "$filename.h5"))
        return joinpath(path, "$filename.h5")
    else
        throw(ArgumentError("No datafile for $filename was found at the given path: $path"))
    end
end

function keystraj(datafile::String)
    if endswith(datafile, ".jld2")
        file = jldopen(datafile, "r")
    elseif endswith(datafile, ".h5")
        file = h5open(datafile, "r")
    end
    keys_traj = keys(file)
    close(file)

    return keys_traj
end

MLUtils.numobs(ds::Dataset) = ds.meta["n_trajectories"]

function MLUtils.getobs!(buffer, ds::Dataset, idx)
    key = ds.meta["keys_trajectories"][idx]

    set_dt!(buffer, ds, key)

    for fn in ds.meta["feature_names"]
        alloc_traj!(buffer, ds, fn)

        match_data = match_keys!(buffer, ds, key, fn)

        set_traj_data!(buffer, match_data, ds, fn)
    end
    set_edges!(buffer, ds, key)

    prepare_trajectory!(buffer, ds.meta, ds.meta["device"])

    buffer["mask"] = Int32.(findall(
        x -> x in ds.meta["types_updated"], buffer["node_type"][1, :, 1])) |>
                     ds.meta["device"]

    buffer["val_mask"] = Float32.(map(
        x -> x in ds.meta["types_updated"], buffer["node_type"][:, :, 1]))
    buffer["val_mask"] = repeat(
        buffer["val_mask"], sum(size(buffer[field], 1)
        for field in ds.meta["target_features"]), 1) |>
                         ds.meta["device"]

    buffer["inflow_mask"] = repeat(buffer["node_type"][:, :, 1] .== 1,
        sum(size(buffer[field], 1) for field in ds.meta["target_features"]), 1) |>
                            ds.meta["device"]

    create_base_graph!(buffer, ds.meta["features"]["node_type"]["data_max"],
        ds.meta["features"]["node_type"]["data_min"], ds.meta["device"])

    return buffer
end

function MLUtils.getobs(ds::Dataset, idx)
    traj_dict = Dict{String, Any}()

    getobs!(traj_dict, ds, idx)

    return traj_dict
end

function set_dt!(traj_dict::Dict{String, Any}, ds::Dataset, key::String)
    local dt

    lock(ds.lock) do
        if endswith(ds.datafile, ".jld2")
            file = jldopen(ds.datafile, "r")
            traj = file[key]
            dt = Float32.(file[key][ds.meta["dt"]])
        else
            file = h5open(ds.datafile, "r")
            traj = open_group(file, key)
            dt = Float32.(Base.read(traj, ds.meta["dt"]))
        end
        traj_dict["dt"] = dt
        close(file)
    end
    if ds.meta["trajectory_length"] == -1
        # trajectory length is inferred from dt of the trajectory
        traj_dict["trajectory_length"] = length(dt)
    else
        traj_dict["trajectory_length"] = ds.meta["trajectory_length"]
    end
end

function alloc_traj!(traj_dict::Dict{String, Any}, ds::Dataset, fn::String)
    dim = haskey(ds.meta["features"][fn], "dim") ? ds.meta["features"][fn]["dim"] : 1
    if ds.meta["features"][fn]["type"] == "static"
        tl = 1
    elseif ds.meta["features"][fn]["type"] == "dynamic"
        tl = traj_dict["trajectory_length"]
    else
        throw(ArgumentError("feature type of feature \"$fn\" must be static or dynamic"))
    end
    if !haskey(traj_dict, fn)
        traj_dict[fn] = zeros(
            getfield(Base, Symbol(uppercasefirst(ds.meta["features"][fn]["dtype"]))),
            dim, prod(ds.meta["dims"]), tl)
    end
    if haskey(ds.meta["features"][fn], "has_ev") && ds.meta["features"][fn]["has_ev"]
        if !haskey(traj_dict, fn * ".ev")
            traj_dict[fn * ".ev"] = zeros(
                eltype(traj_dict[fn]), 2, prod(ds.meta["dims"]), tl)
        end
    end
end

function match_keys!(traj_dict::Dict{String, Any}, ds::Dataset, key::String, fn::String)
    if haskey(ds.meta["features"][fn], "split") && ds.meta["features"][fn]["split"]
        rx = Regex(replace(
            replace(replace(ds.meta["features"][fn]["key"], "[" => "\\["),
                "]" => "\\]"),
            "%d" => "\\d+") * "\\[\\d+\\]")
    else
        rx = Regex(replace(
            replace(replace(ds.meta["features"][fn]["key"], "[" => "\\["),
                "]" => "\\]"),
            "%d" => "\\d+"))
    end

    match_data = Dict()

    lock(ds.lock) do
        if endswith(ds.datafile, ".jld2")
            file = jldopen(ds.datafile, "r")
            traj = file[key]
            rx_match = eachmatch.(rx, keys(traj))
            deleteat!(rx_match, findall(isnothing, rx_match))
            matches = unique(getfield.(rx_match, :match))
            for m in matches
                match_data[m] = traj[m]
                if haskey(ds.meta["features"][fn], "has_ev") &&
                   ds.meta["features"][fn]["has_ev"]
                    match_data[m * ".ev"] = traj[m * ".ev"]
                end
            end
        else
            file = h5open(ds.datafile, "r")
            traj = open_group(file, key)
            rx_match = match.(rx, keys(traj))
            deleteat!(rx_match, findall(isnothing, rx_match))
            matches = unique(getfield.(rx_match, :match))
            for m in matches
                match_data[m] = Base.read(traj, m)
                if haskey(ds.meta["features"][fn], "has_ev") &&
                   ds.meta["features"][fn]["has_ev"]
                    match_data[m * ".ev"] = Base.read(traj, m * ".ev")
                end
            end
        end
        close(file)
    end

    return match_data
end

function set_traj_data!(traj_dict::Dict{String, Any}, match_data, ds::Dataset, fn::String)
    for (m, data) in match_data
        if !occursin("]", m[1:(end - 1)])
            idx = Colon()
            if haskey(ds.meta["features"][fn], "split") &&
               ds.meta["features"][fn]["split"]
                coord = Base.parse.(Int, split(split(m, r"(\[|\])")[2], ","))
            else
                coord = Colon()
            end

            fn_k = occursin(".ev", m) ? "$fn.ev" : fn

            if ds.meta["features"][fn]["type"] == "dynamic"
                if ndims(data) == 2
                    traj_dict[fn_k][coord, :, :] = data[
                        coord, 1:traj_dict["trajectory_length"]]
                else
                    traj_dict[fn_k][coord, :, :] = data[1:traj_dict["trajectory_length"]]
                end
            else
                traj_dict[fn_k][coord, :, :] .= data
            end

        else
            idx = Base.parse.(Int, split(split(m, r"(\[|\])")[2], ","))
            if haskey(ds.meta["features"][fn], "split") &&
               ds.meta["features"][fn]["split"]
                coord = Base.parse.(Int, split(split(m, r"(\[|\])")[4], ","))
            else
                coord = Colon()
            end

            fn_k = occursin(".ev", m) ? "$fn.ev" : fn

            if ds.meta["features"][fn]["type"] == "dynamic"
                if ndims(data) == 2
                    traj_dict[fn_k][coord, dims_to_li(ds.meta["dims"], idx), :] = data[
                        coord, 1:traj_dict["trajectory_length"]]
                else
                    traj_dict[fn_k][coord, dims_to_li(ds.meta["dims"], idx), :] = data[
                        1:traj_dict["trajectory_length"]]
                end
            else
                traj_dict[fn_k][coord, dims_to_li(ds.meta["dims"], idx), :] .= data
            end
        end
    end
end

function set_edges!(traj_dict::Dict{String, Any}, ds::Dataset, key::String)
    lock(ds.lock) do
        if endswith(ds.datafile, ".jld2")
            file = jldopen(ds.datafile, "r")

            if haskey(ds.meta, "custom_edges")
                edges = nothing
                throw(ArgumentError("Custom edge definition is currently not supported for JLD2 files."))
            elseif haskey(ds.meta, "dims")
                edges = create_edges(ds.meta["dims"], traj_dict["node_type"],
                    haskey(meta, "no_edges_node_types") ? meta["no_edges_node_types"] : [])
            else
                throw(ErrorException("No key for edges was provided in metafile. possible keys are [\"dims\", \"custom_edges\"]"))
            end
        else
            file = h5open(ds.datafile, "r")

            if haskey(ds.meta, "custom_edges")
                edges = read_edges(file[key],
                    ds.meta["custom_edges"],
                    traj_dict["node_type"],
                    haskey(ds.meta, "no_edges_node_types") ?
                    ds.meta["no_edges_node_types"] :
                    [],
                    haskey(ds.meta, "exclude_node_indices") ?
                    ds.meta["exclude_node_indices"] : [])
            elseif haskey(ds.meta, "dims")
                edges = create_edges(ds.meta["dims"], traj_dict["node_type"],
                    haskey(ds.meta, "no_edges_node_types") ?
                    ds.meta["no_edges_node_types"] : [])
            else
                throw(ErrorException("No key for edges was provided in metafile. possible keys are [\"dims\", \"custom_edges\"]"))
            end
        end
        close(file)
        traj_dict["edges"] = hcat(sort(edges)...)
    end
end

"""
    create_edges(dims, node_type)

Creates a mesh with the given dimensions

## Arguments
- `dims`: Array with the dimensions of the mesh.
- `node_type`: Array of node types from the data file.
- `excluded_node_types`: Vector of node types that should not be connected with edges.

## Returns
- Vector of connected node pair indices (as vectors).
"""
function create_edges(dims, node_type, no_edges_node_types)
    li = LinearIndices(Tuple(dims))
    edges = Vector{Vector{Int32}}()

    #################################################
    # 1D-Meshes are connected in order by their id  #
    #                                               #
    # 2D-Meshes are not supported yet               #
    #                                               #
    # 3D-Meshes are connected in order by their id, #
    # starting the count from z then y and then x   #
    #################################################
    if length(dims) == 1
        for i in 1:(dims[1] - 1)
            push!(edges, [i, i + 1])
        end
    elseif length(dims) == 2
        throw(ArgumentError("2D-Meshes are not supported yet"))
    elseif length(dims) == 3
        dim_x, dim_y, dim_z = dims

        function add_edge!(edges, x, y, z, cond, shift)
            if cond
                if node_type[1, li[x + shift[1], y + shift[2], z + shift[3]], 1] ∉
                   no_edges_node_types
                    push!(
                        edges, [li[x, y, z], li[x + shift[1], y + shift[2], z + shift[3]]])
                end
            end
        end

        for x in 1:dim_x
            for y in 1:dim_y
                for z in 1:dim_z
                    if node_type[1, li[x, y, z], 1] ∉ no_edges_node_types
                        add_edge!(edges, x, y, z, x != dim_x, [1, 0, 0])
                        add_edge!(edges, x, y, z, y != dim_y, [0, 1, 0])
                        add_edge!(edges, x, y, z, z != dim_z, [0, 0, 1])
                    else
                        if [li[x, y, z], li[x, y, z]] ∉ edges
                            push!(edges, [li[x, y, z], li[x, y, z]])
                        end
                    end
                end
            end
        end
    end

    return edges
end

"""
    read_edges(traj::Group, node_type, no_edges_node_types::Vector{Int}, exclude_node_indices::Vector{Int})

    Read edges from trajectory group.

    ## Arguments

    - `traj`: HDF5 group containing this trajectory's data.
    - `node_type`: Array of node types from the data file.
    - `excluded_node_types`: Vector of node types that should not be connected with edges.
    - `exclude_node_indices`: Vector of node indices that should not be connected with edges.

    ## Returns

    - Vector of connected node pair indices (as vectors).
"""
function read_edges(
        traj::Group, edge_key, node_type, no_edges_node_types, exclude_node_indices)
    if !haskey(traj, edge_key)
        throw(KeyError(
            "Key '$(edge_key)' not found in trajectory group '$(HDF5.name(traj))'"))
    end
    edges = read_dataset(traj, edge_key)
    exclude_indices = findall(x -> x ∈ no_edges_node_types, node_type)
    exclude_indices = vcat(exclude_indices, exclude_node_indices)
    filter!(x -> x[1] ∉ exclude_indices && x[2] ∉ exclude_indices, edges)
    edge_vec = Vector{Vector{Int32}}()
    for edge in edges
        push!(edge_vec, [edge[1], edge[2]])
    end
    return edge_vec
end

"""
    add_targets!(data, fields, device)

Shifts the datapoints beginning from second index back in order to use them as ground truth data (used for derivative based strategies).

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `fields`: Node features of the MGN.
- `device`: Device where the data should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
"""
function add_targets!(data, fields, device)
    new_data = deepcopy(data)
    for (key, value) in data
        if startswith(key, "target|") || key == "dt"
            continue
        end
        if ndims(value) > 2 && size(value)[end] > 1
            if key == "mesh_pos" || key == "node_type" || key == "cells"
                new_data[key] = value[:, :, 1:(end - 1)]
            else
                new_data[key] = device(value[:, :, 1:(end - 1)])
            end
            if key in fields
                new_data["target|" * key] = device(value[:, :, 2:end])
            end
        end
    end
    for (key, value) in new_data
        data[key] = value
    end
end

"""
    preprocess!(data, noise_fields, noise_stddevs, types_noisy, ts, device)

Adds noise to the given features and shuffles the datapoints if a derivative based strategy is used.

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `noise_fields`: Node features to which noise is added.
- `noise_stddevs`: Array of standard deviations of the noise, where the length is either one if broadcasted or equal to the length of features.
- `types_noisy`: Node types to which noise is added.
- `ts`: Training strategy that is used.
- `device`: Device where the data should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).
"""
function preprocess!(data, noise_fields, noise_stddevs, types_noisy, ts, device)
    if length(noise_stddevs) != 1 && length(noise_stddevs) != length(noise_fields)
        throw(DimensionMismatch("dimension of noise must be 1 or match noise fields: noise has dim $(size(noise_stddevs)), noise fields has dim $(size(noise_fields))"))
    end
    for (i, nf) in enumerate(noise_fields)
        d = Normal(0.0f0, length(noise_stddevs) > 1 ? noise_stddevs[i] : noise_stddevs[1])
        noise = rand(d, size(data[nf])) |> device

        mask = findall(x -> x ∉ types_noisy, data["node_type"][1, :, 1])
        noise[:, mask, :] .= 0
        data[nf] += noise
    end

    seed = make_seed(1234)
    rng = MersenneTwister(seed)

    for key in keys(data)
        if key == "edges" || length(data[key]) == 1 || size(data[key])[end] == 1
            continue
        end
        if typeof(ts) <: DerivativeStrategy && ts.random
            if key != "dt"
                data[key] = data[key][repeat([:], ndims(data[key]) - 1)...,
                    shuffle(rng,
                        ts.window_size == 0 ? collect(1:end) : collect(1:(ts.window_size)))]
            end
        end
        seed!(rng, seed)
    end
end

"""
    prepare_trajectory!(data, meta, device; types_noisy, noise_stddevs, ts)

Transfers the data to the given device and configures the data if a derivative based strategy is used.

## Arguments
- `data`: Data from the dataset containing one trajectory.
- `meta`: Metadata of the dataset.
- `device`: Device where the data should be loaded (see [Lux GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management#gpu-management)).

## Keyword Arguments
- `types_noisy`: Node types to which noise is added.
- `noise_stddevs`: Array of standard deviations of the noise, where the length is either one if broadcasted or equal to the length of features.
- `ts`: Training strategy that is used.

## Returns
- Transfered data.
- Metadata of the dataset.
"""
function prepare_trajectory!(data, meta, device::Function)
    if !isnothing(meta["training_strategy"]) &&
       (typeof(meta["training_strategy"]) <: DerivativeStrategy)
        add_targets!(data, meta["target_features"], device)
        preprocess!(data, meta["target_features"], meta["noise_stddevs"],
            meta["types_noisy"], meta["training_strategy"], device)
        for field in meta["feature_names"]
            if field == "mesh_pos" || field == "node_type" || field == "cells" ||
               field in meta["target_features"]
                continue
            end
            data[field] = device(data[field])
        end
    else
        for field in meta["feature_names"]
            if field == "mesh_pos" || field == "node_type" || field == "cells"
                continue
            end
            data[field] = device(data[field])
        end
    end
    return data, meta
end
