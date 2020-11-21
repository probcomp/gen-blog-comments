# example, where we know the ground truth KL divergence..

bernoulli_kl(p, q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))

@gen function model()
    z ~ bernoulli(0.3)
    x ~ bernoulli(z ? 0.1 : 0.8)
end

function posterior_bernoulli_prob(x::Bool)
    if x
        z_true = 0.3 * 0.1
        z_false = 0.7 * 0.8
    else
        z_true = 0.3 * (1 - 0.1)
        z_false = 0.7 * (1 - 0.8)
    end
    return z_true / (z_true + z_false)
end

@gen function approximation(observations)
    z ~ bernoulli(0.6)
end

function run_it(n)
    x_true_symmetric_kls = Float64[]
    x_false_symmetric_kls = Float64[]
    for i in 1:n
        (model_trace, kl_estimate) = simulative_aide(model, (), select(:x), approximation)
        if model_trace[:x]
            push!(x_true_symmetric_kls, kl_estimate)
        else
            push!(x_false_symmetric_kls, kl_estimate)
        end
    end
    x_true_symmetric_kl_estimate = sum(x_true_symmetric_kls) / length(x_true_symmetric_kls)
    x_false_symmetric_kl_estimate = sum(x_false_symmetric_kls) / length(x_false_symmetric_kls)
    return (x_true_symmetric_kl_estimate, x_false_symmetric_kl_estimate)
end

function test_it()
    actual_true, actual_false = test_it(1000000)
    expected_true_prob = posterior_bernoulli_prob(true)
    expected_false_prob = posterior_bernoulli_prob(false)
    expected_true = bernoulli_kl(expected_true_prob, 0.6) + bernoulli_kl(0.6, expected_true_prob)
    expected_false = bernoulli_kl(expected_false_prob, 0.6) + bernoulli_kl(0.6, expected_false_prob)
    println((expected_true, actual_true))
    println((expected_false, actual_false))
    @assert isapprox(expected_true, actual_true, rtol=0.1)
    @assert isapprox(expected_false, actual_false, rtol=0.1)
end

#test_it

