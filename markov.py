import numpy as np
from scipy.stats import norm

def metropolis_probability(delta_e, temperature, prob_parameter=None):
        return min(1, np.exp(-delta_e / temperature))


def exponential_decrease_probability(delta_e, temperature, prob_parameter): 
    return np.exp(-prob_parameter * delta_e / temperature)


def gaussian_probability(delta_e, temperature, prob_parameter): 
    return np.exp(-(delta_e ** 2) / (2 * (prob_parameter ** 2) * temperature))


def simmulated_annealing(energy_function, initial_state, markov_chain_next, init_temperature, n_iter, 
                         temp_scaling_factor = None, temp_decrease_function='linear', prob_function='metropolis', prob_parameter=None):
    if prob_function == 'metropolis':
        acceptance_probability = lambda delta_e, temperature, prob_parameter: metropolis_probability(delta_e, temperature)
    elif prob_function == 'exponential':
        acceptance_probability = lambda delta_e, temperature, prob_parameter: exponential_decrease_probability(delta_e, temperature, prob_parameter)
    elif prob_function == 'gaussian':
        acceptance_probability = lambda delta_e, temperature, prob_parameter: gaussian_probability(delta_e, temperature, prob_parameter)
    else:
        raise ValueError('prob_function must be either "metropolis", "exponential" or "gaussian"')
    
    if temp_decrease_function == 'linear':
        temp_decrease = lambda t, i: init_temperature - i*temp_scaling_factor
        if not temp_scaling_factor:
            temp_scaling_factor = (init_temperature-1e-6)/n_iter #wyliczony tak współczynnik, aby temperatura zeszła do 1e-6 po n_iter iteracjach

    elif temp_decrease_function == 'exponential':
        temp_decrease = lambda t, i: t*temp_scaling_factor
        if not temp_scaling_factor:
            temp_scaling_factor = (1/init_temperature)**(1/n_iter)

    else:
        raise ValueError('temp_decrease_function must be either "linear" or "exponential"')
    
    temperature = init_temperature
    state = markov_chain_next(initial_state)
    for i in range(n_iter):
        temperature = temp_decrease(temperature, i)
        state_new = markov_chain_next(state)
        delta_e = energy_function(state_new) - energy_function(state)
        if delta_e < 0:
            state = state_new
            continue

        if acceptance_probability(delta_e, temperature, prob_parameter) >= np.random.random():
            state = state_new

    return state

def run_experiments(get_new_theta, num_experiments, fun_to_minimize, get_next_element,
                    init_temperature, sim_n_iter, temp_scaling_factor, verbose=False, temp_decrease_function='linear', prob_function='metropolis', prob_parameter=None):
    theta = get_new_theta()
    # print(f"Initial theta: {np.sum(theta)}")
    errors = np.zeros(num_experiments)
    for i in range(num_experiments):
        init_theta = get_new_theta()
        theta_new = simmulated_annealing(fun_to_minimize, init_theta, get_next_element,
                                            init_temperature, sim_n_iter, temp_scaling_factor, temp_decrease_function, prob_function, prob_parameter)
        errors[i] = np.linalg.norm(theta-theta_new)**2
        # print(f'theta: {theta}, theta_new: {theta_new}, error: {errors[i]}')
        if verbose and i % np.ceil(num_experiments / 100) == 0:
            print(f"\rExperiment {i+1}/{num_experiments}", end="")
    if verbose:
        print()
    return errors

def run_sim_part1(m, d, sigma, num_experiments, init_temperature, sim_n_iter, temp_scaling_factor= None,
                   verbose=False, temp_decrease_function='linear', prob_function='metropolis', prob_parameter=None):
    get_new_theta = lambda: np.random.binomial(1, 0.5, d)
    X = np.random.normal(0, 1, (m, d))
    theta = get_new_theta()
    ksi = np.random.normal(0, sigma)
    y = X @ theta + ksi
    function_to_minimize = lambda theta_new: np.linalg.norm(y - (X @ theta_new), 2)

    def get_next_element(theta_old):
        # tu chyba było źle albo to ma robić coś innego niż myślę

        # theta_old[np.random.randint(m)] = 1 - theta_old[np.random.randint(m)]
        # theta_old[np.random.randint(d)] = 1 - theta_old[np.random.randint(d)]
        i = np.random.randint(d)
        theta_old[i] = 1 - theta_old[i]

        return theta_old
    
    errors = run_experiments(get_new_theta, num_experiments, function_to_minimize, get_next_element,
                                init_temperature, sim_n_iter, temp_scaling_factor, verbose, temp_decrease_function, prob_function, prob_parameter)
    return 1/d*np.mean(errors)

def markov_get_next_theta_sparse(theta):
    j = np.random.choice(np.nonzero(theta)[0])
    theta[j] = 0
    i = np.random.choice(np.nonzero(1-theta)[0])
    theta[i] = 1
    return theta

def run_sim_part2(m, d, sigma, s, num_experiments, init_temperature, sim_n_iter, temp_scaling_factor = None,
                  verbose=False, temp_decrease_function='linear', prob_function='metropolis', prob_parameter=None):
    def get_new_theta():
        theta = np.zeros(d)
        theta[np.random.choice(d, s, replace=False)] = 1
        return theta
    
    X = np.random.normal(0, 1, (m, d))
    theta = get_new_theta()
    ksi = np.random.normal(0, sigma)
    y = X @ theta + ksi
    fun_to_minimize = lambda theta_new: np.linalg.norm(y - (X @ theta_new), 2)

    errors = run_experiments(get_new_theta, num_experiments, fun_to_minimize, markov_get_next_theta_sparse,
                                init_temperature, sim_n_iter, temp_scaling_factor, verbose, temp_decrease_function, prob_function, prob_parameter)
    return 1/s*np.mean(errors)

def run_sim_part3(m, d, sigma, s, num_experiments, init_temperature, sim_n_iter, temp_scaling_factor= None,
                  verbose=False, temp_decrease_function='linear', prob_function='metropolis', prob_parameter=None):
    def get_new_theta():
        theta = np.zeros(d)
        theta[np.random.choice(d, s, replace=False)] = 1
        return theta
    
    X = np.random.normal(0, 1, (m, d))
    theta = get_new_theta()
    ksi = np.random.normal(0, sigma)
    y = np.sign(X @ theta + ksi)
    fun_to_minimize = lambda theta_new: -np.sum(np.log(np.clip( norm().cdf(y*(X@theta_new)/sigma), 1e-10, 1-1e-10)))

    errors = run_experiments(get_new_theta, num_experiments, fun_to_minimize, markov_get_next_theta_sparse,
                                init_temperature, sim_n_iter, temp_scaling_factor, verbose, temp_decrease_function, prob_function, prob_parameter)
    return 1/s*np.mean(errors)

