import numpy as np
from scipy.stats import norm

def simmulated_annealing(energy_function, initial_state, markov_chain_next, init_temperature, n_iter, temp_scaling_factor):
    temperature = init_temperature
    state = markov_chain_next(initial_state)
    for _ in range(n_iter):
        temperature *= temp_scaling_factor
        state_new = markov_chain_next(state)
        current_energy, new_energy = energy_function(state), energy_function(state_new)
        acceptance_probability = min(1, np.exp(-1/temperature*(new_energy-current_energy)))
        if acceptance_probability >= np.random.random():
            state = state_new
    return state

def run_experiments(get_new_theta, num_experiments, fun_to_minimize, get_next_element,
                    init_temperature, sim_n_iter, temp_scaling_factor):
    theta = get_new_theta()
    errors = np.zeros(num_experiments)
    for i in range(num_experiments):
        init_theta = get_new_theta()
        theta_new = simmulated_annealing(fun_to_minimize, init_theta, get_next_element,
                                            init_temperature, sim_n_iter, temp_scaling_factor)
        errors[i] = np.linalg.norm(theta-theta_new)**2
    return errors

def run_sim_part1(m, d, sigma, num_experiments, init_temperature, sim_n_iter, temp_scaling_factor):
    get_new_theta = lambda: np.random.binomial(1, 0.5, d)
    X = np.random.normal(0, 1, (m, d))
    theta = get_new_theta()
    ksi = np.random.normal(0, sigma)
    y = X @ theta + ksi
    function_to_minimize = lambda theta_new: np.linalg.norm(y - (X @ theta_new), 2)

    def get_next_element(theta_old):
        theta_old[np.random.randint(m)] = 1 - theta_old[np.random.randint(m)]
        return theta_old
    
    errors = run_experiments(get_new_theta, num_experiments, function_to_minimize, get_next_element,
                                init_temperature, sim_n_iter, temp_scaling_factor)
    return 1/d*np.mean(errors)

def markov_get_next_theta_sparse(theta):
    j = np.random.choice(np.nonzero(theta)[0])
    theta[j] = 0
    i = np.random.choice(np.nonzero(1-theta)[0])
    theta[i] = 1
    return theta

def run_sim_part2(m, d, sigma, s, num_experiments, init_temperature, sim_n_iter, temp_scaling_factor):
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
                                init_temperature, sim_n_iter, temp_scaling_factor)
    return 1/s*np.mean(errors)

def run_sim_part3(m, d, sigma, s, num_experiments, init_temperature, sim_n_iter, temp_scaling_factor):
    def get_new_theta():
        theta = np.zeros(d)
        theta[np.random.choice(d, s, replace=False)] = 1
        return theta
    
    X = np.random.normal(0, 1, (m, d))
    theta = get_new_theta()
    ksi = np.random.normal(0, sigma)
    y = np.sign(X @ theta + ksi)
    fun_to_minimize = lambda theta_new: -np.sum(np.log(norm().cdf(y*(X@theta_new)/sigma)))

    errors = run_experiments(get_new_theta, num_experiments, fun_to_minimize, markov_get_next_theta_sparse,
                                init_temperature, sim_n_iter, temp_scaling_factor)
    return 1/s*np.mean(errors)

if __name__ == "__main__":
    error = run_sim_part3(
        m = 5,
        d = 100,
        sigma = 0.2,
        s = 5,
        num_experiments = 100,
        init_temperature = 200,
        sim_n_iter = 100,
        temp_scaling_factor = 0.9)
    print(error)