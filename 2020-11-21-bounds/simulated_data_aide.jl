using Gen

function simulative_aide(
        model_trace::Trace, # should be a posterior sample, can be obtained through rejection sampling
        observations::Selection,
        fit_posterior::Function, get_approx_args::Function)

    observation_choices = get_selected(get_choices(model_trace), observations)

    # fit the variational approximation
    approximation = fit_posterior(observation_choices)

    # run AIDE
    approximation_trace = simulate(approximation, get_approx_args(observation_choices))
    forward_model_log_weight = get_score(model_trace)
    forward_approximation_log_weight = get_score(approximation_trace)
    (_, backward_model_log_weight) = generate(
        get_gen_fn(model_trace), get_args(model_trace), merge(observation_choices, get_choices(approximation_trace)))
    (_, backward_approximation_log_weight) = generate(
        approximation, get_approx_args(observation_choices), get_choices(model_trace))
    single_sample_symmetric_kl_estimate = (
        (forward_model_log_weight - backward_approximation_log_weight) + # KL from posterior to approximation
        (forward_approximation_log_weight - backward_model_log_weight)) # KL from approximation to posterior
    return (
        single_sample_symmetric_kl_estimate,
        forward_model_log_weight, backward_approximation_log_weight,
        forward_approximation_log_weight, backward_model_log_weight, approximation_trace)
end


function simulative_aide(
        model::GenerativeFunction, model_args::Tuple,
        observations::Selection,
        fit_posterior::Function, get_approx_args::Function)

    # simulate latent and observed data from the model
    model_trace = simulate(model, model_args)
    observation_choices = get_selected(get_choices(model_trace), observations)

    # fit the variational approximation
    approximation = fit_posterior(observation_choices)

    # run AIDE
    approximation_trace = simulate(approximation, get_approx_args(observation_choices))
    forward_model_log_weight = get_score(model_trace)
    forward_approximation_log_weight = get_score(approximation_trace)
    (_, backward_model_log_weight) = generate(
        model, model_args, merge(observation_choices, get_choices(approximation_trace)))
    (_, backward_approximation_log_weight) = generate(
        approximation, get_approx_args(observation_choices), get_choices(model_trace))
    single_sample_symmetric_kl_estimate = (
        (forward_model_log_weight - backward_approximation_log_weight) +
        (forward_approximation_log_weight - backward_model_log_weight))
    return (
        single_sample_symmetric_kl_estimate,
        forward_model_log_weight, backward_approximation_log_weight,
        forward_approximation_log_weight, backward_model_log_weight, approximation_trace)
end

# interestingly, the 'gap' may differ between observed data sets...
# save the rest for a later blog post

#function simulative_aide(
        #model::GenerativeFunction, model_args::Tuple,
        #observations::Selection,
        #approximation::GenerativeFunction,
        #get_approx_args::Function)
    #return simulative_aide(
        #model, model_args, observations,
        #(_) -> approximation, get_approx_args)
##end
