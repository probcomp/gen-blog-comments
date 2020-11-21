using Gen
using PyPlot
import Distributions
import Random
import SpecialFunctions
using JLD
using Flux.Optimise
using GenFluxOptimizers
using Printf: @sprintf
PyPlot.matplotlib.use("Agg")

##########################
# von mises distribution #
##########################

struct VonMises <: Distribution{Float64} end

function Gen.logpdf(::VonMises, x::Real, mu::Real, k::Real)
    (mu < -pi || mu > pi || x < -pi || x > pi) && error("out of range")
    lpdf = k * (cos(x - mu) - 1) - log(SpecialFunctions.besselix(0.0, k)) - log(2*pi)
    @assert !isinf(lpdf) && !isnan(lpdf)
    return lpdf
end

Gen.logpdf_grad(::VonMises, x, mu, k) = (nothing, nothing, nothing)
Gen.has_argument_grads(::VonMises) = (false, false)
Gen.has_output_grad(::VonMises) = false

function Gen.random(::VonMises, mu::Real, k::Real)
    (mu < -pi || mu > pi) && error("out of range")
    x = rand(Distributions.VonMises(0.0, k)) + mu
    x = x - floor(x / (2*pi)) * 2 * pi # between 0 and 2 * pi
    if x > pi
        x = x - 2*pi
    end
    @assert (-pi <= x <= pi)
    return x # between -pi and pi
end

(::VonMises)(mu, k) = random(VonMises(), mu, k)

const von_mises = VonMises()

####################
# generative model #
####################

@gen function heading_model()
    x ~ normal(1.0, 1.0)
    y ~ normal(0.0, 1.0)
    heading = atan(y, x)
    measured_heading ~ von_mises(heading, 100.0)
    return heading
end

###################################
# black box variational inference #
###################################

@gen function q_axis_aligned_gaussian()
    @param x_mu::Float64
    @param y_mu::Float64
    @param x_log_std::Float64
    @param y_log_std::Float64
    x ~ normal(x_mu, exp(x_log_std))
    y ~ normal(y_mu, exp(y_log_std))
end

sigmoid(x) = 1 ./ (1 .+ exp.(-x))

@gen function q_amortized(heading)
    @param b1::Vector{Float64}
    @param W1::Matrix{Float64}
    @param b2::Vector{Float64}
    @param W2::Matrix{Float64}
    @param b3::Vector{Float64}
    @param W3::Matrix{Float64}
    hidden1 = sigmoid.(b1 .+ (W1 * [heading, cos(heading), sin(heading)]))
    hidden2 = sigmoid.(b2 .+ (W2 * hidden1))
    output = b3 .+ (W3 * hidden2)
    x ~ normal(output[1], exp(output[2]))
    y ~ normal(output[3] + (output[4] * x), exp(output[5] + (output[6] * x)))
end

function data_generator()
    trace = simulate(heading_model, ())
    return (((trace[:measured_heading],), get_choices(trace)))
end

function train_q_amortized()
    adam_update = ParamUpdate(FluxOptimConf(Optimise.ADAM, (0.1, (0.9, 0.999))), q_amortized => [:b1, :W1, :b2, :W2, :b3, :W3])
    num_hidden1 = 4
    num_hidden2 = 4
    num_output = 6
    init_param!(q_amortized, :b1, zeros(num_hidden1))
    init_param!(q_amortized, :W1, randn(num_hidden1, 3) * 0.1)
    init_param!(q_amortized, :b2, zeros(num_hidden2))
    init_param!(q_amortized, :W2, randn(num_hidden2, num_hidden1) * 0.1)
    init_param!(q_amortized, :b3, zeros(num_output))
    init_param!(q_amortized, :W3, randn(num_output, num_hidden2) * 0.1)
    epoch_size = 100000
    minibatch_size = 100
    epoch_inputs = Vector{Tuple}(undef, epoch_size)
    epoch_choice_maps = Vector{ChoiceMap}(undef, epoch_size)    
    for i=1:epoch_size
        (epoch_inputs[i], epoch_choice_maps[i]) = data_generator()
    end
    objs = Float64[]
    for iter in 1:1000
        permuted = Random.randperm(epoch_size)
        minibatch_idx = permuted[1:minibatch_size]
        minibatch_inputs = epoch_inputs[minibatch_idx]
        minibatch_choice_maps = epoch_choice_maps[minibatch_idx]
        total_weight = 0.0
        for (inputs, constraints) in zip(minibatch_inputs, minibatch_choice_maps)
            (trace, weight) = generate(q_amortized, inputs, constraints)
            total_weight += weight
            accumulate_param_gradients!(trace)
        end
        obj = total_weight / minibatch_size
        push!(objs, obj)
        println("iter $iter, obj: $(obj)")
        apply!(adam_update)
    end

    # plot the training
    figure()
    plot(objs)
    xlabel("iterations")
    ylabel("objective estimate")
    savefig("training.png")

    # save params to disk
    save("params.jld", Dict(
        String(name) => get_param(q_amortized, name)
        for name in [:b1, :W1, :b2, :W2, :b3, :W3]))
end


function load_q_amortized_params()
    params = load("params.jld")
    println("loading params for q_amortized, got: $(keys(params))")
    for name in [:b1, :W1, :b2, :W2, :b3, :W3]
        set_param!(q_amortized, name, params[String(name)])
    end
end

function nn_inference(measured_heading::Float64, num_samples::Int)

    # return samples from the approximation, and the ELBO estimate
    traces = []
    elbo_ests = Float64[]
    for i in 1:num_samples
        trace = simulate(q_amortized, (measured_heading,))
        push!(traces, trace)
        _, model_score = generate(heading_model, (), merge(get_choices(trace), choicemap((:measured_heading, measured_heading))))
        push!(elbo_ests, model_score - get_score(trace))
        
    end
    xs = Float64[trace[:x] for trace in traces]
    ys = Float64[trace[:y] for trace in traces]
    return (xs, ys, sum(elbo_ests)/length(elbo_ests))
end

function bbvi_inference(
        q, measured_heading::Float64, num_samples::Int, num_iters::Int;
        samples_per_iter=100, verbose=false, step_size=0.001, step_size_beta=100)

    # initialize parameters
    init_param!(q, :x_mu, cos(measured_heading))
    init_param!(q, :y_mu, sin(measured_heading))
    init_param!(q, :x_log_std, 0.0)
    init_param!(q, :y_log_std, 0.0)

    # fit parameters using BBVI
    update = ParamUpdate(GradientDescent(step_size, step_size_beta), q)
    (elbo_est, _, elbo_history) = black_box_vi!(
        heading_model, (),
        choicemap((:measured_heading, measured_heading)),
        q, (), update;
        iters=num_iters, samples_per_iter=samples_per_iter, verbose=verbose)
    x_mu = get_param(q, :x_mu)
    y_mu = get_param(q, :y_mu)
    x_log_std = get_param(q, :x_log_std)
    y_log_std = get_param(q, :y_log_std)

    # return samples from the approximation, and the ELBO estimate
    traces = []
    for i in 1:num_samples
        push!(traces, simulate(q, ()))
    end
    xs = Float64[trace[:x] for trace in traces]
    ys = Float64[trace[:y] for trace in traces]
    return (xs, ys, elbo_est, elbo_history)
end

#########################################
# vanilla importance sampling inference #
#########################################

function vanilla_importance_sampling(measured_heading::Float64, num_importance_samples::Int, num_samples::Int)
    xs = Float64[]
    ys = Float64[]
    (traces, log_normalized_weights, lml_estimate) = importance_sampling(
        heading_model, (), choicemap((:measured_heading, measured_heading)), num_importance_samples)
    weights = exp.(log_normalized_weights)
    traces = [traces[categorical(weights)] for i in 1:num_samples]
    xs = Float64[trace[:x] for trace in traces]
    ys = Float64[trace[:y] for trace in traces]
    return (xs, ys, lml_estimate)
end

##############################################################
# stochastic upper bound via conditional importance sampling #
##############################################################

function cis_stochastic_upper_bound(trace, observed::Selection, num_samples::Int)
    model = get_gen_fn(trace)
    model_args = get_args(trace)
    observations = get_selected(get_choices(trace), observed)
    log_weights = Vector{Float64}(undef, num_samples)
    log_weights[1] = project(trace, observed)
    for i=2:num_samples
        (_, log_weights[i]) = generate(model, model_args, observations)
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    return log_ml_estimate
end

######################
# plotting utilities #
######################

function rect(x, y, w, h; kw_args...)
    gca().add_patch(matplotlib.patches.Rectangle((x-(w/2), y-(h/2)), w, h; kw_args...))
end

function draw_robot()
    width = 0.2
    length = 0.5
    rect(-length/3.5, -width/2, 0.15, 0.15; color="gray")
    rect(-length/3.5, width/2, 0.15, 0.15; color="gray")
    rect(length/3.5, width/2, 0.15, 0.15; color="gray")
    rect(length/3.5, -width/2, 0.15, 0.15; color="gray")
    rect(0, 0, length, width; color="black")
end

function draw_heading(measured_heading::Float64; kw_args...)
    xmeasured_heading = 10 * cos(measured_heading)
    ymeasured_heading = 10 * sin(measured_heading)
    plot([0, xmeasured_heading], [0, ymeasured_heading]; kw_args...)
end

function set_bounds()
    gca().set_xlim((-3, 3))
    gca().set_ylim((-3, 3))
    gca().set_xticks([-2, 0, 2])
    gca().set_yticks([-2, 0, 2])
end

###############
# experiments #
###############

function generate_grid_results()
    Random.seed!(1)
    num_posterior_samples = 400
    measured_headings = collect(range(-pi, pi, length=200))

    # amortized inference
    println("nn...")
    nn_results = []
    for (i, measured_heading) in enumerate(measured_headings)
        xs, ys, elbo_est = nn_inference(measured_heading, num_posterior_samples)
        push!(nn_results, Dict(
            "elbo_est" => elbo_est,
            "measured_heading" => measured_heading,
            "xs" => xs,
            "ys" => ys))
    end

    # bbvi
    println("bbvi...")
    bbvi_results = Dict()
    for num_iters in [10, 500]
        bbvi_results[num_iters] = []
        for (i, measured_heading) in enumerate(measured_headings)
            xs, ys, elbo_est = bbvi_inference(
                q_axis_aligned_gaussian, measured_heading, num_posterior_samples, num_iters;
                samples_per_iter=30, step_size=0.005, step_size_beta=100)
            push!(bbvi_results[num_iters], Dict(
                "measured_heading" => measured_heading,
                "xs" => xs,
                "ys" => ys,
                "elbo_est" => elbo_est))
        end
    end

    # vanilla importance sampling
    println("importance sampling...")
    vis_results = Dict()
    for num_importance_samples in [10, 100, 1000, 10000]
        vis_results[num_importance_samples] = []
        for (i, measured_heading) in enumerate(measured_headings)
            xs, ys, log_marginal_likelihood_est = vanilla_importance_sampling(
                measured_heading, num_importance_samples, num_posterior_samples)
            push!(vis_results[num_importance_samples], Dict(
                "measured_heading" => measured_heading,
                "xs" => xs,
                "ys" => ys,
                "log_marginal_likelihood_est" => log_marginal_likelihood_est))
        end
    end

    results = Dict(
        "nn_results" => nn_results,
        "bbvi_results" => bbvi_results,
        "importance_sampling_results" => vis_results)

    return results
end


include("simulated_data_aide.jl")

function bbvi_fit_posterior(observations)
    bbvi_inference(q_axis_aligned_gaussian, observations[:measured_heading], 0, 500;
                samples_per_iter=30, step_size=0.005, step_size_beta=100)
    return q_axis_aligned_gaussian
end

function generate_simulative_aide_binned_results()

    num_bins = 64
    get_bin(heading) = Int(ceil((heading + pi) / (pi/32)))
    bin_to_heading(bin) = (bin-1) * (pi/32) - pi

    println("generating simulated data...")
    traces_by_bin = [[] for bin in 1:num_bins]
    Random.seed!(1)
    num_required = 12000
    cur = 1
    @time while minimum(map(length, traces_by_bin)) < num_required
        if (cur % 1000) == 0
            println(map(length, traces_by_bin))
        end
        cur += 1
        trace = simulate(heading_model, ())
        bin = get_bin(trace[:measured_heading])
        if length(traces_by_bin[bin]) < num_required
            push!(traces_by_bin[bin], trace)
        end
    end

    # NN
    println("running AIDE for NN...")
    nn_estimates = Float64[]
    nn_headings = Float64[]
    nn_estimates_by_bin = [Float64[] for bin in 1:num_bins]
    for bin in 1:num_bins
        @time for trace in traces_by_bin[bin]
            estimate, pq_forward, pq_backward, qp_forward, qp_backward, q_trace = simulative_aide(
                trace, select(:measured_heading),
                (observations) -> q_amortized, (observations) -> (observations[:measured_heading],))
            push!(nn_estimates_by_bin[bin], estimate)
            push!(nn_estimates, estimate)
            push!(nn_headings, trace[:measured_heading])
        end
    end

    # BBVI
    println("running AIDE for BBVI...")
    bbvi_estimates = Float64[]
    bbvi_headings = Float64[]
    bbvi_estimates_by_bin = [Float64[] for bin in 1:num_bins]
    for bin in 1:num_bins
        @time for trace in traces_by_bin[bin][1:100]
            estimate, _, _ = simulative_aide(
                trace, select(:measured_heading),
                bbvi_fit_posterior, (observations) -> ())
            push!(bbvi_estimates_by_bin[bin], estimate)
            push!(bbvi_estimates, estimate)
            push!(bbvi_headings, trace[:measured_heading])
        end
    end

    headings_for_bins = Float64[bin_to_heading(bin) for bin in 1:num_bins]
    return Dict(
        "headings_for_bins" => headings_for_bins,
        "bbvi_headings" => bbvi_headings,
        "bbvi_estimates" => bbvi_estimates,
        "bbvi_estimates_by_bin" => bbvi_estimates_by_bin,
        "nn_headings" => nn_headings,
        "nn_estimates" => nn_estimates,
        "nn_estimates_by_bin" => nn_estimates_by_bin)
end


##################
# generate plots #
##################

function generate_plots_prior()

    # prior on location of object
    close("all")
    Random.seed!(1)
    figure(figsize=(2, 2), dpi=400)
    xs = []
    ys = []
    for i in 1:200
        trace = simulate(heading_model, ())
        push!(xs, trace[:x])
        push!(ys, trace[:y])
    end
    scatter(xs, ys, color="red", alpha=0.2, s=5)
    draw_robot()
    set_bounds()
    savefig("prior-overlay.png")

    # prior on location of object and simulated measurement
    close("all")
    Random.seed!(1)
    figure(figsize=(6, 3), dpi=200)
    for i in 1:8
        subplot(2, 4, i)
        trace = simulate(heading_model, ())
        scatter([trace[:x]], [trace[:y]], color="red", alpha=1.0, s=20, linestyle="-", linewidth=1)
        draw_heading(trace[:measured_heading]; color="blue", linestyle="--", linewidth=2)
        draw_heading(trace[]; color="blue")
        draw_robot()
        set_bounds()
    end
    tight_layout()
    savefig("prior.png")

    close("all")
end

function generate_plots_posterior(grid_results)

    grid_indices = map((x) -> Int(floor(x)),range(1, 200, length=16))

    figsize=(12, 12)
    dpi=250

    close("all")
    figure(figsize=figsize, dpi=dpi)
    for (i, grid_idx) in enumerate(grid_indices)
        subplot(4, 4, i)
        bbvi_results = grid_results["bbvi_results"][500][grid_idx]
        measured_heading = bbvi_results["measured_heading"]
        xs = bbvi_results["xs"]
        ys = bbvi_results["ys"]
        draw_robot()
        draw_heading(measured_heading; color="blue", linestyle="--", linewidth=2)
        scatter(xs, ys, color="red", alpha=0.2, s=10)
        set_bounds()
    end
    tight_layout()
    savefig("approx-posterior-bbvi.png")

    close("all")
    figure(figsize=figsize, dpi=dpi)
    for (i, grid_idx) in enumerate(grid_indices)
        subplot(4, 4, i)
        vis_results = grid_results["importance_sampling_results"][10000][grid_idx]
        measured_heading = vis_results["measured_heading"]
        xs = vis_results["xs"]
        ys = vis_results["ys"]
        draw_robot()
        draw_heading(measured_heading; color="blue", linestyle="--", linewidth=2)
        scatter(xs, ys, color="red", alpha=0.2, s=10)
        set_bounds()
    end
    tight_layout()
    savefig("approx-posterior-vis.png")

    close("all")
    figure(figsize=figsize, dpi=dpi)
    for (i, grid_idx) in enumerate(grid_indices)
        subplot(4, 4, i)
        nn_results = grid_results["nn_results"][grid_idx]
        measured_heading = nn_results["measured_heading"]
        xs = nn_results["xs"]
        ys = nn_results["ys"]
        draw_robot()
        draw_heading(measured_heading; color="blue", linestyle="--", linewidth=2)
        scatter(xs, ys, color="red", alpha=0.2, s=10)
        set_bounds()
    end
    tight_layout()
    savefig("approx-posterior-nn.png")
end

function generate_plots_grid_lml(grid_results)

    # show just ELBO estimates from BBVI, and from the neural net
    bbvi_measured_headings = Float64[]
    elbo_ests = Float64[]
    for result in grid_results["bbvi_results"][500]
        push!(elbo_ests, result["elbo_est"])
        push!(bbvi_measured_headings, result["measured_heading"])
    end
    nn_measured_headings = Float64[]
    nn_elbo_ests = Float64[]
    for result in grid_results["nn_results"]
        push!(nn_elbo_ests, result["elbo_est"])
        push!(nn_measured_headings, result["measured_heading"])
    end

    close("all")
    figure(figsize=(6, 2), dpi=200)
    scatter(bbvi_measured_headings, elbo_ests, s=20, alpha=0.5, color="teal")
    ylabel("ELBO estimate")
    xlabel("measured heading")
    gca().set_xlim((-pi, pi))
    tight_layout()
    savefig("bbvi-elbos.png")

    # also show estimates using importance sampling with 1000 particles, and the estimated KL divergences on the right
    for num_importance_samples in [10, 100, 1000, 10000]
        vis_measured_headings = Float64[]
        lml_ests = Float64[]
        for result in grid_results["importance_sampling_results"][num_importance_samples]
            push!(lml_ests, result["log_marginal_likelihood_est"])
            push!(vis_measured_headings, result["measured_heading"])
        end
        @assert vis_measured_headings == bbvi_measured_headings
        @assert vis_measured_headings == nn_measured_headings
        bbvi_kls = lml_ests .- elbo_ests
        nn_kls = lml_ests .- nn_elbo_ests
    
        close("all")
        figure(figsize=(12, 3), dpi=200)
        subplot(1, 2, 1)
        scatter(bbvi_measured_headings, elbo_ests, s=20, alpha=0.5, label="BBVI ELBO estimate", color="teal")
        scatter(vis_measured_headings, lml_ests, label="IS LML estimate", s=20, alpha=0.5, color="orange")
        legend()
        xlabel("measured heading")
        ylabel("ELBO and LML estimates")
        gca().set_xlim((-pi, pi))
        subplot(1, 2, 2)
        scatter(vis_measured_headings, bbvi_kls, s=20, alpha=0.5, color="teal", label="BBVI KL estimate")
        legend()
        ylabel("KL divergence est.")
        xlabel("measured heading")
        gca().set_xlim((-pi, pi))
        # rarely, the BBVI optimization fails, rescale so we focus on the interesing KL regime
        gca().set_ylim((gca().get_ylim()[1], 3)) 
        tight_layout()
        savefig("bbvi-vis-$num_importance_samples.png")
    end
end

function plot_binned_aide_results(results, show_bbvi, show_nn, show_raw_data, show_averages, ylim, fname)

    headings_for_bins = results["headings_for_bins"]
    bbvi_estimates_by_bin = results["bbvi_estimates_by_bin"]
    bbvi_averages = map((arr) -> sum(arr)/length(arr), bbvi_estimates_by_bin)
    nn_estimates_by_bin = results["nn_estimates_by_bin"]
    nn_averages = map((arr) -> sum(arr)/length(arr), nn_estimates_by_bin)

    close("all")
    if show_nn
        figure(figsize=(6, 4), dpi=200)
    else
        figure(figsize=(6, 2), dpi=200)
    end

    bin_width = pi/32
    show_both = show_nn && show_bbvi
    if show_nn
        if show_averages
            for (i, (x, y)) in enumerate(zip(headings_for_bins, nn_averages))
                plot([x-(bin_width/2), x+(bin_width/2)], (y, y), color="purple",
                    label=(i == 1 && show_both ? "NN" : nothing))
            end
        end
        if show_raw_data
            scatter(results["nn_headings"], results["nn_estimates"], color="purple", s=1, alpha=0.1,
                label=(show_both ? "NN" : nothing))
        end
    end
    if show_bbvi
        if show_averages
            for (i, (x, y)) in enumerate(zip(headings_for_bins, bbvi_averages))
                plot([x-(bin_width/2), x+(bin_width/2)], (y, y), color="teal",
                    label=(i == 1 && show_both ? "BBVI" : nothing))
            end
        end
        if show_raw_data
            scatter(results["bbvi_headings"], results["bbvi_estimates"], color="teal", s=2, alpha=0.5,
                label=(show_both ? "BBVI" : nothing))
        end
    end
    if show_both
        legend()
    end
    xlabel("measured heading")
    ylabel("Symmetric KL divergence est.")
    gca().set_xlim((-pi, pi))
    gca().set_ylim(ylim)
    savefig(fname)
end

function generate_binning_animation()
    num_bins = 64
    get_bin(heading) = Int(ceil((heading + pi) / (pi/32)))
    bin_to_heading(bin) = (bin-1) * (pi/32) - pi

    traces_by_bin = [[] for bin in 1:num_bins]
    Random.seed!(1)
    num_required = 100
    println("generating simulated data...")
    cur = 1
    animation_idx = 1
    xs = map(bin_to_heading, 1:num_bins)
    @time while minimum(map(length, traces_by_bin)) < num_required
        if (cur % 1000) == 0
            figure(figsize=(6, 3))
            plot([-pi, pi], [num_required, num_required], linestyle="--", color="black")
            bar(xs, map(length, traces_by_bin), color="black", width=pi/32)
            title("after $cur simulations")
            xlabel("measured heading (radians)")
            ylabel("number of samples")
            tight_layout()
            savefig(@sprintf("binning_animation/%03d.png", animation_idx))
            close("all")
            animation_idx += 1
        end
        cur += 1
        trace = simulate(heading_model, ())
        bin = get_bin(trace[:measured_heading])
        if length(traces_by_bin[bin]) < num_required
            push!(traces_by_bin[bin], trace)
        end
    end
end

##################
# occluder model #
##################

struct Occluder
    x::Float64
    ymin::Float64
    ymax::Float64
end

function occluded(x, y, occluder::Occluder)
    @assert occluder.ymin < occluder.ymax
    y_proj = (y / x) * occluder.x
    return (x > occluder.x) && (occluder.ymin < y_proj < occluder.ymax)
end

@gen function occluder_heading_model(k, occluders)
    x ~ normal(1.0, 1.0)
    y ~ normal(0.0, 1.0)
    theta = atan(y, x)
    if any([occluded(x, y, occ) for occ in occluders])
        # it is occluded, but may be false positive
        is_detection ~ bernoulli(0.1)
        if is_detection
            measured_heading ~ von_mises(0.0, 0.00001)
        end
    else
        # it is occluded, but may be false negative
        is_detection ~ bernoulli(0.90)
        if is_detection
            measured_heading ~ von_mises(theta, k)
        end
    end
end

function measurement_to_choicemap(measurement)
    obs = choicemap()
    if measurement === nothing
        obs[:is_detection] = false
    else
        obs[:is_detection] = true
        obs[:measured_heading] = measurement
    end
    return obs
end

function show_occluder_posterior(measurement, occluders)
    obs_choices = measurement_to_choicemap(measurement)
    (traces, log_normalized_weights, lml_estimate) = importance_sampling(
        occluder_heading_model, (100.0, occluders), obs_choices, 10000)
    weights = exp.(log_normalized_weights)
    traces = [traces[categorical(weights)] for i in 1:500]
    xs = Float64[trace[:x] for trace in traces]
    ys = Float64[trace[:y] for trace in traces]

    scatter(xs, ys, color="red", alpha=0.3, s=10)
    for occluder in occluders
        plot([occluder.x, occluder.x], [occluder.ymin, occluder.ymax], linewidth=3, color="black")
    end
    if obs_choices[:is_detection]
        draw_heading(obs_choices[:measured_heading]; color="blue")
        title("posterior for detection heading: $(@sprintf("%03.2f", obs_choices[:measured_heading]/pi))*pi")
    else
        title("posterior for no detection")
    end
    draw_robot()
    set_bounds()
end

function occlusion_example()

    figure()
    show_occluder_posterior(nothing, [Occluder(0.8, 0.5, 2.0), Occluder(0.3, -2.0, -1.0)])
    tight_layout()
    savefig("occlusion-no-detection.png")
    close("all")

    figure(figsize=(32, 8), dpi=100)
    for (i, measurement) in enumerate(range(-pi, stop=pi, length=16))
        subplot(2, 8, i)
        show_occluder_posterior(measurement, [Occluder(0.8, 0.5, 2.0), Occluder(0.3, -2.0, -1.0)])
    end
    tight_layout()
    savefig("occlusion-detections.png")
    close("all")
end


###############
# entry-point #
###############

train_q_amortized()
load_q_amortized_params()

grid_results = generate_grid_results()
save("grid_results.jld", "grid_results", grid_results)
grid_results = load("grid_results.jld")["grid_results"]
generate_plots_posterior(grid_results)
generate_plots_grid_lml(grid_results)

binned_aide_results = generate_simulative_aide_binned_results()
save("simulative_aide_binned_results.jld", binned_aide_results)
binned_aide_results = load("simulative_aide_binned_results.jld")

plot_binned_aide_results(binned_aide_results, true, false, true, false, (-8, 25), "binned_kl_estimates_bbvi_only_raw_data.png")
plot_binned_aide_results(binned_aide_results, true, false, false, true, (0, 8), "binned_kl_estimates_bbvi_only_averages.png")
plot_binned_aide_results(binned_aide_results, false, true, true, false, (-5, 200), "binned_kl_estimates_nn_only_raw_data.png")
plot_binned_aide_results(binned_aide_results, true, true, false, true, (0, 40), "binned_kl_estimates_averages.png")

#generate_binning_animation()
#occlusion_example()

close("all")
