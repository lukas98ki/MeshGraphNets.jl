#
# Copyright (c) 2023 Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import SciMLBase: AbstractSensitivityAlgorithm, ODEFunction
import SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP
using ComponentArrays: ComponentArray, ComponentVector

#######################################################
# Abstract type and functions for training strategies #
#######################################################

abstract type TrainingStrategy end

"""
    prepare_training(strategy)

Function that is executed once before training. Can be overwritten by training strategies if necessary.

## Arguments
- `strategy`: Used training strategy.

## Returns
- Tuple containing the results of the function.
"""
function prepare_training(::TrainingStrategy)
    return (nothing,)
end

"""
    get_delta(strategy, trajectory_length)

Returns the delta between samples in the training data.

## Arguments
- `strategy`: Used training strategy.
- Trajectory length (used for derivative based strategies).

## Returns
- Delta between samples in the training data.
"""
function get_delta(strategy::TrainingStrategy, ::Integer)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    init_train_step(strategy, t, ta)

Function that is executed before each training sample.

## Arguments
- `strategy`: Used training strategy.
- `t`: Tuple containing the variables necessary for initializing training.
- `ta`: Tuple with additional variables that is returned from [prepare_training](@ref).

## Returns
- Tuple containing variables needed for [train_step](@ref).
"""
function init_train_step(strategy::TrainingStrategy, ::Tuple, ::Tuple)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    train_step(strategy, t)

Performs a single training step and return the resulting gradients and loss.

## Arguments
- `strategy`: Solver strategy that is used for training.
- `t`: Tuple that is returned from [`init_train_step`](@ref).

## Returns
- Gradients for optimization step.
- Loss for optimization step.
"""
function train_step(strategy::TrainingStrategy, ::Tuple)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    validation_step(strategy, t)

Performs validation of a single trajectory. Should be overwritten by training strategies to determine simulation and data interval before calling the inner function [_validation_step](@ref).

## Arguments
- `strategy`: Type of training strategy (used for dispatch).
- `t`: Tuple containing the variables necessary for validation.

## Returns
- See [_validation_step](@ref).
"""
function validation_step(strategy::TrainingStrategy, ::Tuple)
    throw(ArgumentError("Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies."))
end

"""
    _validation_step(t, sim_interval, data_interval)

Inner function for validation of a single trajectory.

## Arguments
- `t`: Tuple containing the variables necessary for validation.
- `sim_interval`: Interval that determines the simulated time for the validation.
- `data_interval`: Interval that determines the indices of the timesteps in ground truth and prediction data.

## Returns
- Loss calculated on the difference between ground truth and prediction (via mse).
- Ground truth data with `data_interval` as timesteps.
- Prediction data with `data_interval` as timesteps.
"""
function _validation_step(t::Tuple, sim_interval, data_interval)
    mgn, data, meta, _, solver, solver_dt, fields, node_type, edge_features, senders, receivers, mask, val_mask, inflow_mask, pr = t

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end
    target_fields = meta["target_features"]
    target_edge_fields = intersect(target_fields, meta["edge_features"])
    target_node_fields = intersect(target_fields, meta["feature_names"])
    if length(target_node_fields) != 0 && length(target_edge_fields) != 0
        gt_node = vcat([data[tf] for tf in target_node_fields]...)[:, :, data_interval]
        gt_edge = vcat([data[tf] for tf in target_edge_fields]...)[:, :, data_interval]
        sol_u, _ = rollout(
            solver, mgn, data, fields, meta, target_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, inflow_mask,
            sim_interval[1], sim_interval[end], solver_dt, sim_interval, pr)

        n_timesteps = length(sol_u)
        node_preds_unshaped = [sol_u[t].node for t in 1:n_timesteps]
        node_preds = cat(node_preds_unshaped...; dims = 3)

        edge_preds_unshaped = [sol_u[t].edge for t in 1:n_timesteps]
        edge_preds = cat(edge_preds_unshaped...; dims = 3)

        println("mean gt_node: ", mean(gt_node))
        println("mean node_preds: ", mean(node_preds))
        println("mean gt_edge: ", mean(gt_edge))
        println("mean edge_preds: ", mean(edge_preds))

        error_node = mean((node_preds - gt_node) .^ 2; dims = 3)
        error_edge = mean((edge_preds - gt_edge) .^ 2; dims = 3)
        println("error node: ", mean(error_node[mask]), " error edge: ", mean(error_edge))
        return mean(error_node[mask]) + mean(error_edge)
    elseif length(target_node_fields) != 0
        gt_node = vcat([data[tf] for tf in target_node_fields]...)[:, :, data_interval]
        sol_u, _ = rollout(
            solver, mgn, data, fields, meta, target_node_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, inflow_mask,
            sim_interval[1], sim_interval[end], solver_dt, sim_interval, pr)
        prediction = cat(sol_u...; dims = 3)[:, :, data_interval]

        error = mean((prediction - gt_node) .^ 2; dims = 3)
        # Todo: error for edge

        return mean(error[mask])

    elseif length(target_edge_fields) != 0
        gt_edge = vcat([data[tf] for tf in target_edge_fields]...)[:, :, data_interval]
        sol_u, _ = rollout(
            solver, mgn, data, fields, meta, target_edge_fields, target_dict,
            node_type, edge_features, senders, receivers, val_mask, inflow_mask,
            sim_interval[1], sim_interval[end], solver_dt, sim_interval, pr)
        prediction = cat(sol_u...; dims = 3)[:, :, data_interval]

        error = mean((prediction - gt_edge) .^ 2; dims = 3)
        # Todo: error for edge

        return mean(error)
    else
        throw(ArgumentError("No target features found!"))
    end

    # sol_u, _ = rollout(
    #     solver, mgn, data, fields, meta, meta["target_features"], target_dict,
    #     node_type, edge_features, senders, receivers, val_mask, inflow_mask,
    #     sim_interval[1], sim_interval[end], solver_dt, sim_interval, pr)

    prediction = cat(sol_u...; dims = 3)[:, :, data_interval]

    error = mean((prediction - gt_node) .^ 2; dims = 3)
    # Todo: error for edge

    return mean(error[mask])
end

####################################################################
# Abstract type and functions for solver based training strategies #
####################################################################

abstract type SolverStrategy <: TrainingStrategy end

function get_delta(::SolverStrategy, ::Integer)
    return 1
end

function init_train_step(::SolverStrategy, t::Tuple, ta::Tuple)
    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, _, idx_mask, val_mask = t

    target_edge_fields = intersect(target_fields, meta["edge_features"])
    target_node_fields = intersect(target_fields, meta["feature_names"])

    edge_fields = meta["edge_features"]
    all_fields = vcat(fields, edge_fields)

    target_dict = Dict{String, Int32}()
    for tf in meta["target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end
    # Mixed input array -> diffentiate later between node and edge features
    inputs = Dict{String, AbstractArray}(
        [typeof(data[field]) <: AbstractArray ? (field, data[field][:, :, 1]) :
         (field, data[field]) for field in all_fields]
    )
    # define gt if target fields are given, else init empty array
    gt_node = isempty(target_node_fields) ? Array{Float32}(undef, 0, 0, 0) :
              vcat([data[tf] for tf in target_node_fields]...)
    gt_edge = isempty(target_edge_fields) ? Array{Float32}(undef, 0, 0, 0) :
              vcat([data[tf] for tf in target_edge_fields]...)

    u0_node = size(gt_node, 3) > 0 ? gt_node[:, :, 1] : Array{Float32}(undef, 0, 0)
    u0_edge = size(gt_edge, 3) > 0 ? gt_edge[:, :, 1] : Array{Float32}(undef, 0, 0)

    if !isempty(target_node_fields) && !isempty(target_edge_fields)
        u0 = ComponentArray(; node = copy(u0_node), edge = copy(u0_edge))
        gt = ComponentArray(;
            node = copy(gt_node), edge = copy(gt_edge))
    elseif !isempty(target_node_fields)
        u0 = u0_node
        gt = gt_node
    elseif !isempty(target_edge_fields)
        u0 = u0_edge
        gt = gt_edge
    else
        throw(ArgumentError("No target features found!"))
    end

    return (mgn, data, inputs, fields, meta, target_fields, target_dict,
        node_type, edge_features, senders, receivers, idx_mask, val_mask, u0, gt)
end

function train_step(strategy::SolverStrategy, t::Tuple)
    mgn, data, inputs, fields, meta, target_fields, target_dict, node_type, edge_features, senders, receivers, idx_mask, val_mask, u0, gt = t
    pr = ProgressUnknown(; desc = "Solver progress: ", showspeed = true)
    re = nothing
    if typeof(mgn.model) <: Flux.Chain
        mgn.ps, re = Flux.destructure(mgn.model)
    end
    target_node_fields = intersect(target_fields, meta["feature_names"])
    target_edge_fields = intersect(target_fields, meta["edge_features"])

    ff = ODEFunction{false}((x, p, t) -> ode_func_train(x,
        (mgn, p, re, data, inputs, fields, meta,
            target_fields, target_dict, node_type,
            edge_features, senders, receivers, val_mask, data["inflow_mask"],
            strategy, pr, target_node_fields, target_edge_fields),
        t))

    prob = ODEProblem(ff, u0, (strategy.tstart, strategy.tstop), mgn.ps)

    shoot_loss, shoot_gs = Zygote.withgradient(
        ps -> train_loss(strategy,
            (prob, ps, u0, nothing, gt, val_mask, mgn, target_fields,
                [meta["features"][tf]["dim"] for tf in target_fields], target_node_fields, target_edge_fields)),
        mgn.ps)
    return shoot_gs, shoot_loss
end

"""
    train_loss(strategy, t)

Inner function for a solver based training step that calculates the loss based on the difference between the ground truth and the predicted solution.

## Arguments
- `strategy`: Solver strategy that is used for training.
- `t`: Tuple containing all variables necessary for loss calculation.

## Returns
- Calculated loss.
"""
function train_loss(strategy::SolverStrategy, ::Tuple)
    throw(ArgumentError("Unknown solver based training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available solver strategies."))
end

function validation_step(strategy::SolverStrategy, t::Tuple)
    sim_interval = (strategy.tstart):(strategy.dt):(strategy.tstop)
    data_interval = 1:length(sim_interval)  # Todo: hardcoded adjustment for initial 5 steps instead of one
    return _validation_step(t, sim_interval, data_interval)
end

"""
    SolverTraining(tstart, dt, tstop, solver; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP()), solargs...)

The default solver based training that is normally used for NeuralODEs.
Simulates the system from `tstart` to `tstop` and calculates the loss based on the difference between the prediction and the ground truth at the timesteps `tstart:dt:tstop`.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense = InterpolatingAdjoint(autojacvec = ZygoteVJP())`: The sensitivity algorithm that is used for caluclating the sensitivities.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct SolverTraining <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    solargs::Any
end

function SolverTraining(tstart::Float32,
        dt::Float32,
        tstop::Float32,
        solver::OrdinaryDiffEqAlgorithm;
        sense::AbstractSensitivityAlgorithm = InterpolatingAdjoint(;
            autojacvec = ZygoteVJP(), checkpointing = true),
        solargs...)
    SolverTraining(tstart, dt, tstop, solver, sense, solargs)
end

function train_loss_in_serialization(strategy::SolverTraining, t::Tuple)
    prob, ps, u0, callback_solve, gt, idx_mask, val_mask, n_norm, target_fields, target_dims = t

    sol = solve(remake(prob; p = ps), strategy.solver; u0 = u0,
        saveat = (strategy.tstart):(strategy.dt):(strategy.tstop),
        tstops = (strategy.tstart):(strategy.dt):(strategy.tstop),
        sensealg = strategy.sense, callback = callback_solve, strategy.solargs...)

    pred = typeof(gt) <: CuArray ? CuArray(sol) : Array(sol)

    error = (gt[:, idx_mask, 1:size(pred, 3)] .- pred) .^ 2
    return mean(error)
end

function train_loss(strategy::SolverTraining, t::Tuple)
    prob, ps, u0, callback_solve, gt, val_mask, mgn, target_fields, target_dims, target_node_fields, target_edge_fields = t

    sol = solve(remake(prob; p = ps), strategy.solver; u0 = u0,
        saveat = (strategy.tstart):(strategy.dt):(strategy.tstop),
        tstops = (strategy.tstart):(strategy.dt):(strategy.tstop),
        sensealg = strategy.sense, callback = callback_solve, strategy.solargs...)

    if length(target_node_fields) != 0 && length(target_edge_fields) != 0
        n_timesteps = length(sol)

        n_node_features, n_nodes = size(gt.node)[1:2]
        n_edge_features, n_edges = size(gt.edge)[1:2]

        node_preds_unshaped = [sol[t].node for t in 1:n_timesteps]
        node_preds = cat(node_preds_unshaped...; dims = 3)   # shape: (n_node_features, n_nodes, n_timesteps)

        edge_preds_unshaped = [sol[t].edge for t in 1:n_timesteps]
        edge_preds = cat(edge_preds_unshaped...; dims = 3)   # shape: (n_edge_features, n_edges, n_timesteps)

        normed_pred_n = cat(
            [mgn.n_norm[target_node_fields[i]](node_preds[i, :, :])
             for i in 1:n_node_features]...;
            dims = 1
        )
        normed_gt_n = cat(
            [mgn.n_norm[target_node_fields[i]](gt.node[i, :, :])
             for i in 1:n_node_features]...;
            dims = 1
        )

        normed_pred_e = cat(
            [mgn.e_norm[target_edge_fields[i]](edge_preds[i, :, :])
             for i in 1:n_edge_features]...;
            dims = 1
        )
        normed_gt_e = cat(
            [mgn.e_norm[target_edge_fields[i]](gt.edge[i, :, :])
             for i in 1:n_edge_features]...;
            dims = 1
        )

        error_n = (normed_pred_n .- normed_gt_n) .^ 2 |> cpu_device()
        error_e = (normed_pred_e .- normed_gt_e) .^ 2 |> cpu_device()

        vm = cpu_device()(val_mask)
        masked_error_n = error_n .* vm'

        loss = mean(copy(masked_error_n)) + mean(copy(error_e))
    else
        pred = typeof(gt) <: CuArray ? CuArray(sol) : Array(sol)

        local gt_n
        local pred_n
        println("type gt: ", typeof(gt))
        for i in eachindex(target_fields)
            if length(target_node_fields) != 0
                pred_n = vcat([mgn.n_norm[target_fields[i]](pred[
                                   (sum(target_dims[1:(i - 1)]) + 1):sum(target_dims[1:i]), :, :])
                               for i in eachindex(target_fields)]...)
                gt_n = vcat([mgn.n_norm[target_fields[i]](gt[
                                 (sum(target_dims[1:(i - 1)]) + 1):sum(target_dims[1:i]),
                                 :, 1:size(pred, 3)]) for i in eachindex(target_fields)]...)

            else
                gt_n = vcat([mgn.e_norm[target_fields[i]](gt[
                                 (sum(target_dims[1:(i - 1)]) + 1):sum(target_dims[1:i]),
                                 :, 1:size(pred, 3)]) for i in eachindex(target_fields)]...)
                pred_n = vcat([mgn.e_norm[target_fields[i]](pred[
                                   (sum(target_dims[1:(i - 1)]) + 1):sum(target_dims[1:i]), :, :])
                               for i in eachindex(target_fields)]...)
            end
        end

        error = (gt_n[:, :, 1:size(pred, 3)] .- pred_n) .^ 2 |> cpu_device()

        if length(target_node_fields) != 0
            err_buf = Zygote.Buffer(error)

            vm = cpu_device()(val_mask)

            err_buf[:, :, :] = error
            for i in axes(err_buf, 3)
                err_buf[:, :, i] = err_buf[:, :, i] .* vm
            end
            loss = mean(copy(err_buf))
        end
    end

    # println("Debugs")
    # println("mean node_preds: ", mean(node_preds))
    # println("mean edge_preds: ", mean(edge_preds))
    # println("mean normed_pred_n: ", mean(normed_pred_n))
    # println("mean normed_gt_n: ", mean(normed_gt_n))
    # println("mean normed_pred_e: ", mean(normed_pred_e))
    # println("mean error_n: ", mean(masked_error_n))
    # println("mean error_e: ", mean(error_e))
    # println("loss: ", loss)

    return loss
end

"""
    MultipleShooting(tstart, dt, tstop, solver, interval_size, continuity_term = 100; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true), solargs...)

Similar to SolverTraining, but splits the trajectory into intervals that are solved independently and then combines them for loss calculation.
Useful if the network tends to get stuck in a local minimum if SolverTraining is used.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)`: The sensitivity algorithm that is used for caluclating the sensitivities.
- `interval_size`: Size of the intervals (i.e. number of datapoints in one interval).
- `continuity_term = 100`: Factor by which the error between points of concurrent intervals is multiplied.
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct MultipleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    interval_size::Integer                  # Number of observations in one interval
    continuity_term::Integer
    solargs::Any
end

function MultipleShooting(tstart::Float32,
        dt::Float32,
        tstop::Float32,
        solver::OrdinaryDiffEqAlgorithm;
        sense::AbstractSensitivityAlgorithm = InterpolatingAdjoint(;
            autojacvec = ZygoteVJP(), checkpointing = true),
        interval_size,
        continuity_term = 100,
        solargs...)
    MultipleShooting(
        tstart, dt, tstop, solver, sense, interval_size, continuity_term, solargs)
end

function train_loss(strategy::MultipleShooting, t::Tuple)
    prob, ps, _, callback_solve, gt, val_mask, _, _, _ = t

    tsteps = (strategy.tstart):(strategy.dt):(strategy.tstop)
    ranges = [i:min(length(tsteps), i + strategy.interval_size - 1)
              for i in 1:(strategy.interval_size - 1):(length(tsteps) - 1)]

    sols = [solve(
                remake(
                    prob;
                    p = ps,
                    tspan = (tsteps[first(rg)], tsteps[last(rg)]),
                    u0 = gt[:, :, first(rg)]
                ),
                strategy.solver;
                saveat = tsteps[rg],
                sensealg = strategy.sense,
                callback = callback_solve,
                strategy.solargs...
            ) for rg in ranges]
    group_predictions = typeof(gt) <: CuArray ? CuArray.(sols) : Array.(sols)

    retcodes = [sol.retcode for sol in sols]
    if any(retcodes .!= :Success)
        return Inf
    end

    vm = cpu_device()(val_mask)

    loss = 0
    for (i, rg) in enumerate(ranges)
        error = (gt[:, :, rg] - group_predictions[i]) .^ 2 |> cpu_device()

        err_buf = Zygote.Buffer(error)
        err_buf[:, :, :] = error
        for i in axes(err_buf, 3)
            err_buf[:, :, i] = err_buf[:, :, i] .* vm
        end
        loss += mean(copy(err_buf))

        if i > 1
            loss += strategy.continuity_term *
                    sum(abs, group_predictions[i - 1][:, :, end] - gt[:, :, first(rg)])
        end
    end

    return loss
end

########################################################################
# Abstract type and functions for derivative based training strategies #
########################################################################

abstract type DerivativeStrategy <: TrainingStrategy end

function get_delta(strategy::DerivativeStrategy, trajectory_length::Integer)
    return strategy.window_size > 0 ? strategy.window_size : trajectory_length - 1
end

function init_train_step(::DerivativeStrategy, t::Tuple, ::Tuple)
    mgn, data, meta, fields, target_fields, node_type, edge_features, senders, receivers, datapoint, mask, _ = t

    target_edge_fields = intersect(target_fields, meta["edge_features"])
    target_node_fields = intersect(target_fields, meta["feature_names"])

    # Currently output is not normed in loss function -> so here no norming either
    target_quantities_change_node = vcat([mgn.o_norm[field]((data["target|" * field][
                                              :, :, datapoint] -
                                                             data[field][:, :, datapoint]) /
                                                            (data["dt"][datapoint + 1] -
                                                             data["dt"][datapoint]))
                                          for field in target_node_fields]...)

    target_quantities_change_edge = vcat([mgn.o_norm[field]((data["target|" * field][
                                              :, :, datapoint] -
                                                             data[field][:, :, datapoint]) /
                                                            (data["dt"][datapoint + 1] -
                                                             data["dt"][datapoint]))
                                          for field in target_edge_fields]...)

    # target_quantities_change_node = vcat([((data["target|" * field][
    #                                           :, :, datapoint] -
    #                                         data[field][:, :, datapoint]) /
    #                                        (data["dt"][datapoint + 1] -
    #                                         data["dt"][datapoint]))
    #                                       for field in target_node_fields]...)

    # target_quantities_change_edge = vcat([((data["target|" * field][
    #                                           :, :, datapoint] -
    #                                         data[field][:, :, datapoint]) /
    #                                        (data["dt"][datapoint + 1] -
    #                                         data["dt"][datapoint]))
    #                                       for field in target_edge_fields]...)

    target_quantities_change = (target_quantities_change_node,
        target_quantities_change_edge)

    graph = build_graph(
        mgn, data, fields, datapoint, node_type, edge_features,
        meta["edge_features"], senders, receivers)

    return (mgn, graph, target_quantities_change, mask)
end

function train_step(::DerivativeStrategy, t::Tuple)
    mgn, graph, target_quantities_change, mask = t

    return step!(mgn, graph, target_quantities_change, mask, mse_reduce)
end

function validation_step(::DerivativeStrategy, t::Tuple)
    sim_interval = t[2]["dt"][1]:(t[2]["dt"][2] - t[2]["dt"][1]):t[2]["dt"][t[4]]
    data_interval = 1:t[4]
    return _validation_step(t, sim_interval, data_interval)
end

"""
    DerivativeTraining(; window_size = 0)

Compares the prediction of the system with the derivative from the data (via finite differences).
Useful for initial training of the system since it it faster than training with a solver.

## Keyword Arguments
- `window_size = 0`: Number of steps from each trajectory (starting at the beginning) that are used for training. If the number is zero then the whole trajectory is used.
- `random = true`: Whether the derivatives of the data should shuffled before the training.
"""
struct DerivativeTraining <: DerivativeStrategy
    window_size::Integer
    random::Bool
end
function DerivativeTraining(; window_size::Integer = 0, random = true)
    DerivativeTraining(window_size, random)
end
