"""
This library defines a function that runs simulated annealing for the QUBO problem using PyTorch.
Here, each step in the annealing is performed by swapping a random bit, and temperature is in units of energy.
"""

import torch
from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.random import randint


@dataclass
class AnnealingRun:
    """
    Container object defining the annealing run
    """

    beta_vals: ArrayLike
    energy_vals: ArrayLike
    states: ArrayLike
    num_accepts: ArrayLike
    num_rejects: ArrayLike
    Q: ArrayLike
    device: str


def metropolis(
        Q: ArrayLike,
        num_steps: int,
        high_temp: float,
        low_temp: float,
        callback: callable = None,
        init: ArrayLike = None
) -> AnnealingRun:
    """
    Function performing simulated annealing

    :param Q: Square matrix in objective function x.T @ (Q @ x)
    :param num_steps: Number of annealing steps
    :param high_temp: High temperature to start annealing at, units of energy
    :param low_temp: Low temperature to end annealing at, units of energy
    :param callback: Optional callback one can use to pass beta, energy pairs as a string, useful if printing to a file mid-run for example, defaults to None
    :param init: Optional initial value for x, defaults to None

    :returns: Returns an AnnealingRun instance with run information
    :raises ValueError: if Q is not a square matrix
    """

    # try to find cuda, if not, use cpu
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if Q.shape[0] != Q.shape[1] or len(Q.shape) != 2:
        raise ValueError(f'Q needs to be a square matrix but has shape {Q.shape}')

    # initialize device and run attributes, sending attributes to appropriate device
    device = torch.device(device_name)
    Q_ = torch.Tensor(Q).to(device)
    betas = torch.linspace(1.0 / high_temp, 1.0 / low_temp, num_steps).to(device)
    energy_vals = torch.zeros(num_steps).to(device)
    states = torch.zeros((num_steps, Q.shape[0])).to(device)
    accepts = torch.zeros(num_steps).to(device)
    rejects = torch.zeros(num_steps).to(device)

    # if initial state isn't provided, create a random vector of 0's and 1's, 50% of each
    if init is None:
        state = (torch.rand(size=(Q.shape[0],)) < 0.5).float().to(device)
    else:
        state = torch.Tensor(init).to(device)

    # calculate initial energy
    energy = torch.dot(state, Q_ @ state)

    if callback:
        callback('beta,energy')

    # generate random numbers now instead of mid-loop
    random_numbers = torch.rand(num_steps).to(device)

    for step in range(num_steps):

        # store energy and state
        energy_vals[step] = energy
        states[step, :] = state
        if callback:
            callback(f'{betas[step]},{energy}')

        # get random integer, and new state from flipping the bit at that random integer
        i = randint(Q.shape[0])

        new_state = state.clone()
        new_state[i] = float(not new_state[i])

        # calculate changes in energy
        change = new_state.T @ Q_ @ new_state - state.T @ Q_ @ state

        # get boltzmann factor corresponding to change
        boltzmann_factor = torch.exp(-betas[step] * change)

        # accept or reject using Metropolis scheme
        if random_numbers[step] < boltzmann_factor:
            state = new_state
            energy += change
            accepts[step] = 1

        else:
            rejects[step] = 1

    # get cumulative number of acceptances and rejections to return
    num_accepts, num_rejects = torch.cumsum(accepts, dim=0), torch.cumsum(rejects, dim=0)

    # return AnnealingRun instance with information
    annealing_attributes = {
        'beta_vals': betas.cpu().numpy(),
        'energy_vals': energy_vals.cpu().numpy(),
        'states': states.cpu().numpy(),
        'num_accepts': num_accepts.cpu().numpy(),
        'num_rejects': num_rejects.cpu().numpy(),
        'Q': Q,
        'device': device_name
    }

    return AnnealingRun(**annealing_attributes)
