 cloudhacks-2024

How to run it:

Download the dependencies in the requirements.txt file and run main.py

'bandit.py' : multi-arm neural bandit model using a neural network with experience replay for dynamic strategy selection.

'ddos_env.py' : define DDoS mitigation env. using OpenAI Gym Framework with my optimization strategies as actions.

'ddos.py' : simulates a DDoS attack using asynchronous HTTP requests.

'ec.py' : short for evolutionary computation, where I use my Differential Evolution Optimization Algorithm.

'ga.py' : short for genetic algorithm, where I implement my Genetic Algorithm Optimization Algorithm

'si.py' : short for swarm intelligence, where I define the Reinforcement Learning-based Swarm Optimization (RSO) Algorithm.

'main.py' : starts the simulation of DDoS attacks and how my mitigation strategies work against it. It also evaluates the overall performance and plots the results.

## Key Components ##

'bandit.py'
Neural Network: Defines a simple feedforward neural network with dropout layers.

Experience Replay: Implements experience replay to store and sample past experiences for training.

NeuralBandit: Combines the neural network and experience replay to select the best mitigation strategy dynamically.

'ddos_env.py'

DDoSMitigationEnv: Custom environment with three actions corresponding to different optimization strategies (Differential Evolution, Swarm Intelligence, Genetic Algorithm).

'ddos.py'

Async HTTP Requests: Uses aiohttp to send asynchronous HTTP requests, simulating a DDoS attack.

Simulation Orchestration: Manages the simulation of DDoS attacks and logs the results.

'ec.py'

Differential Evolution: Implements the Differential Evolution algorithm to optimize parameters for server performance.

'ga.py'

Genetic Algorithm: Implements a Genetic Algorithm using the DEAP library for parameter optimization.

'si.py'

Swarm Optimization: Implements a custom Reinforcement Learning-based Swarm Optimization algorithm to find the best solution for server performance.


## Conclusion ##

I have integrated advanced AI techniques and optimization algorithms to provide a novel and adaptive solution for DDoS attack prevention. By leveraging continuous learning and dynamic strategy selection, it offers a robust and scalable approach to mitigating evolving cyber threats.