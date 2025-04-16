#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import ProgressMeter: ProgressUnknown

import ChainRulesCore: @ignore_derivatives

"""
    rollout(solver, mgn, initial_state, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, data, start, stop, dt, saves; show_progress = true)

Solves the ODEProblem of the MGN with the given solver.

## Arguments
- `solver`: Solver that is used for evaluating the system.
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `initial_state`: Initial state of the system.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `inflow_mask`: Vector of indices of nodes that are defined as inflow nodes.
- `data`: Simulation data used for setting the inputs on the inflow nodes.
- `start`: Start time of the simulation.
- `stop`: Stop time of the simulation.
- `dt`: If set, the solver will use fixed timesteps.
- `saves`: Timesteps where the solution is saved at.

## Keyword Arguments
- `show_progress = true`: Whether a progress bar should be displayed.

## Returns
- Solution of the ODEProblem at the specified timesteps.
- Timesteps corresponding to the solution.
"""
function rollout(solver, mgn::GraphNetwork, data, fields, meta, target_fields,
        target_dict, node_type, edge_features, senders, receivers, val_mask,
        inflow_mask, start, stop, dt, saves, pr = nothing)
    interval = (start, stop)
    x0 = vcat([typeof(data[field]) <: AbstractArray ? data[field][:, :, 5] :
               data[field] for field in target_fields]...)
    inputs = Dict{String, AbstractArray}(
        [typeof(data[field]) <: AbstractArray ? (field, data[field][:, :, 1:5]) :
         (field, data[field]) for field in fields]
    )
    # inputs = reshape(inputs, :, size(inputs, 2))
    re = nothing
    if typeof(mgn.model) <: Flux.Chain
        mgn.ps, re = Flux.destructure(mgn.model)
    end
    prob = ODEProblem(ode_func_eval, x0, interval,
        (mgn, mgn.ps, re, data, inputs, fields, meta, target_fields,
            target_dict, node_type, edge_features, senders, receivers,
            val_mask, inflow_mask, saves[2] - saves[1], pr))

    if isnothing(dt)
        sol = solve(prob, solver; saveat = saves, tstops = saves)
    else
        sol = solve(prob, solver; adaptive = false, dt = dt, saveat = saves)
    end

    if !isnothing(pr)
        finish!(pr)
    end

    return sol.u, sol.t
end

"""
    ode_func_train(x, (mgn, ps, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, strategy, pr), t)

Inner function for training the system via solver.

## Arguments
- `x`: Current state of the system.
- Tuple containing variables needed for a step of the ODE.
- `t`: Current timestep of the system.

The parameter tuple contains the following variables:
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ps`: Parameters of the network inside the MGN.
- `data`: Simulation data used for setting the inputs on the inflow nodes.
- `inputs`: Dictionary of the initial state without the target features.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `inflow_mask`: Vector of indices of nodes that are defined as inflow nodes.
- `strategy`: Training strategy used for training.
- `pr`: Progress bar for logging.

## Returns
- See [ode_step](@ref).
"""
# Todo: Entfernen des x in input zu tun. wird in ode_step gemacht
function ode_func_train(x,
        (mgn, ps, re, data, inputs, fields, meta, target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, inflow_mask, strategy, pr),
        t)
    # bx = Zygote.Buffer(x)
    # bx[:, :, end][inflow_mask] = vcat([data[field][:, :, floor(Int, t / strategy.dt) + 1]
    #                                    for field in target_fields]...)[inflow_mask]

    # bx = Zygote.Buffer(x)
    # bx[:, :] = x
    # bx[inflow_mask] = vcat([data[field][:, :, floor(Int, t / strategy.dt) + 1]
    #                         for field in target_fields]...)[inflow_mask]

    # new_values = vcat([data[field][:, :, floor(Int, t / strategy.dt) + 1]
    #                    for field in target_fields]...)

    # bx[:, :, end][inflow_mask] .= new_values[inflow_mask]

    ############# new test
    # new_inputs = deepcopy(inputs)

    for k in target_fields
        if (ndims(inputs[k]) == 3)
            inputs[k] = vcat(eachslice(inputs[k]; dims = 3)...)
        end

        # inputs[k][1:(end - 1), :] = inputs[k][2:end, :]

        # inputs[k][end, :] = x[:]

        inputs[k] = vcat(inputs[k][2:end, :], x)
    end

    # for k in target_fields
    #     if (ndims(inputs[k]) == 3)
    #         # Flatten von 3D auf 2D
    #         inputs_2d = vcat(eachslice(inputs[k]; dims = 3)...)
    #     else
    #         inputs_2d = inputs[k]
    #     end

    #     shifted = inputs_2d[2:end, :]
    #     new_input_k = vcat(shifted, x[:]')  # x[:] ist (n,) --> transponiert (1, n)

    #     inputs[k] = new_input_k
    # end

    return ode_step(x,
        (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, pr),
        t)

    # return ode_step(bx,
    #     (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
    #         node_type, edge_features, senders, receivers, val_mask, pr),
    #     t)
end

"""
    ode_func_eval(x, (mgn, ps, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, inflow_mask, saves_dt, pr), t)

Inner function for evaluating the ODEProblem.

## Arguments
- `x`: Current state of the system.
- Tuple containing variables needed for a step of the ODE.
- `t`: Current timestep of the system.

The parameter tuple contains the following variables:
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ps`: Parameters of the network inside the MGN.
- `data`: Simulation data used for setting the inputs on the inflow nodes.
- `inputs`: Dictionary of the initial state without the target features.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `inflow_mask`: Vector of indices of nodes that are defined as inflow nodes.
- `saves_dt`: Timesteps where the input of the system is updated.
- `pr`: Progress bar for logging.

## Returns
- See [ode_step](@ref).
"""
function ode_func_eval(x,
        (mgn, ps, re, data, inputs, fields, meta, target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, inflow_mask, saves_dt, pr),
        t)
    # x[inflow_mask] = vcat([data[field][:, :, floor(Int, t / saves_dt) + 1]
    #                        for field in target_fields]...)[inflow_mask]

    # return ode_step(x,
    #     (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
    #         node_type, edge_features, senders, receivers, val_mask, pr),
    #     t)

    for k in target_fields
        if (ndims(inputs[k]) == 3)
            # Flatten von 3D auf 2D
            inputs_2d = vcat(eachslice(inputs[k]; dims = 3)...)
        else
            inputs_2d = inputs[k]
        end

        shifted = inputs_2d[2:end, :]
        new_input_k = vcat(shifted, x[:]')  # x[:] ist (n,), transponiert zu (1, n)

        inputs[k] = new_input_k
    end

    return ode_step(x,
        (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, pr),
        t)
end

"""
    ode_step(x, (mgn, ps, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, val_mask, pr), t)

Performs a single step of the ODEProblem (see [ode_func_train](@ref) and [ode_func_eval](@ref)).

## Arguments
- `x`: Current state of the system.
- Tuple containing variables needed for a step of the ODE.
- `t`: Current timestep of the system.

The parameter tuple contains the following variables:
- `mgn`: [GraphNetwork](@ref) that should be evaluated.
- `ps`: Parameters of the network inside the MGN.
- `inputs`: Dictionary of the initial state without the target features.
- `fields`: Node features of the MGN.
- `meta`: Metadata of the dataset.
- `target_fields`: Output features of the MGN.
- `target_dict`: Dictionary containing the output features and their dimensions as key-value pair.
- `node_type`: Onehot vector of the node types used for node features.
- `edge_features`: Array of edge features for each edge in the graph.
- `senders`: Vector of indices where each edge in the graph starts.
- `receivers`: Vector of indices where each edge in the graph ends.
- `val_mask`: Bitmask specifying which nodes should be updated.
- `pr`: Progress bar for logging.

## Returns
- Output of the ODE at the current timestep.
"""
function ode_step(x,
        (mgn, ps, re, inputs, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, pr),
        t)
    # new_inputs = deepcopy(inputs)  # oder copy(), je nach Bedarf und Typ
    offset = 1
    # for k in target_fields
    #     features = target_dict[k]
    #     nodes = size(x, 2)

    #     # Hole das neue Feature und reshape es auf (features, nodes, 1)
    #     new_value = reshape(
    #         x[offset:(offset + features - 1), :],
    #         features, nodes, 1
    #     )

    #     # Verschiebe den "Zeitschritt-Speicher" und füge das neue sample an
    #     # Inputs[k] ist erwartungsgemäß (features, nodes, timesteps)
    #     # Wir verschieben: [:, :, 2:end] → entfernt den ältesten Zeitschritt
    #     new_inputs[k] = cat(
    #         new_inputs[k][:, :, 2:end],  # alte Werte ab zweitem timestep
    #         new_value;                   # neue Werte
    #         dims = 3                     # concateniere entlang der Zeitschritt-Achse
    #     )

    #     offset += features
    # end

    # Todo: datapoint was hardcoded 1, now adjusted to 5 cause of looking at more states
    # Todo: inputs=data and data should be a trajectory, but here it is only targetfeature->nodefeature for single timestep???

    graph = build_graph(
        mgn, inputs, fields, 5, node_type, edge_features, senders, receivers)
    if isnothing(re)
        output, st = mgn.model(graph, ps, mgn.st)
        mgn.st = st
    else
        output = re(ps)(graph)
    end

    # println("output: ", output)
    # println("output size: ", size(output))
    # readline()

    return output .* val_mask

    # indices = [meta["features"][tf]["dim"] for tf in target_fields]

    # buf = Zygote.Buffer(output)
    # for i in eachindex(target_fields)
    #     buf[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :] = inverse_data(
    #         mgn.o_norm[target_fields[i]],
    #         output[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :])
    # end

    # @ignore_derivatives begin
    #     if !isnothing(pr)
    #         next!(pr; showvalues = [(:t, "$(t)")])
    #     end
    # end

    # return copy(buf) .* val_mask
end
