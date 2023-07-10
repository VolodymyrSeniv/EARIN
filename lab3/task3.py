
import numpy as np
import matplotlib.pyplot as plt
import time


def rosenbrock(x, y):

    return (1 - x)**2 + 100 * (y - x**2)**2

def mutate(x, y, sigma, mu_prob):

    if np.random.rand() < mu_prob:
        dx, dy = sigma * np.random.randn(2)
        x += dx
        y += dy
    return x, y 

def one_plus_one_rosenbrock(x_min, x_max, y_min, y_max, sigma, mu_prob, n_generations):

    # Initialize a random point
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    f = rosenbrock(x, y)
    x_progress = [x]
    y_progress = [y]
    # Visualization setup
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(x_min, x_max, x_max*2), np.linspace(y_min, y_max, y_max*2))
    Z = rosenbrock(X, Y)
    ax.contour(X, Y, Z, 50, cmap="coolwarm")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.title("1+1 Evolutionary Algorithm on Rosenbrock Function")
    
    # Main loop
    start_time = time.time()
    for i in range(n_generations):
        # Mutate the current point
        x_prime, y_prime = mutate(x, y, sigma, mu_prob)
        
        # Evaluate the fitness of the mutated point
        f_prime = rosenbrock(x_prime, y_prime)
        
        # Select the better point
        if f_prime < f:
            x, y = x_prime, y_prime
            f = f_prime
        
        x_progress.append(x)
        y_progress.append(y)

    end_time = time.time()
    for i in range(len(x_progress)):
         if i % 10 == 0:
            ax.plot(x_progress[i], y_progress[i], "ko", markersize=2)
            plt.pause(0.01)

    
    # Calculate the optimization time
    optimization_time = end_time - start_time
    
    return x, y, f, optimization_time


x_min = int(input("Provide the x_min(provide any integer): "))
x_max = int(input("Provide the x_max(provide any integer): "))
y_min = int(input("Provide the y_min(provide any integer): "))
y_max = int(input("Provide the y_max(provide any integer): "))
sigma = float(input("Provide the mutation strength: "))
mu_prob = float(input("Provide the muttion probability: "))
n_generations = int(input("Provide the number fo generations: "))
# Run the algorithm
x_opt, y_opt, f_opt, optimization_time = one_plus_one_rosenbrock(x_min, x_max, y_min, y_max, sigma, mu_prob, n_generations)

# Print the optimal solution, fitness, and optimization time
print("Optimal solution: x={}, y={}, f={}".format(x_opt, y_opt, f_opt))
print("Optimization time: {} seconds".format(optimization_time))
plt.show()