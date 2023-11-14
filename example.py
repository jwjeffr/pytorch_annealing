#!/usr/bin/python

"""
Small script with a use-case example of anneal.py
"""

from anneal import metropolis
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time


def example(callback):

    # define matrix Q, length x length with random elements between -1/2 and 1/2
    length = 1_000
    num_steps = 200_000
    Q = np.random.uniform(low=-0.5, high=0.5, size=(length, length))

    # define annealing arguments, track time to run
    kwargs = {
        'Q': Q,
        'num_steps': num_steps,
        'high_temp': 10.0,
        'low_temp': 1.0,
        'callback': callback,
        'init': np.random.choice([0.0, 1.0], size=length),
    }

    # perform run
    past = time.perf_counter()
    run = metropolis(**kwargs)
    present = time.perf_counter()
    sec_elapsed = present - past
    print(f"metropolis run with {num_steps:.0f} steps took {sec_elapsed:.3f} seconds on device {run.device}")

    # if callback specified, results are already printed by callback
    if callback:
        return

    # if callback not specified, plot results
    plt.plot(run.beta_vals, run.energy_vals)
    plt.axhline(run.states[-1].T @ run.Q @ run.states[-1], color='black', linestyle=':')
    plt.grid()
    plt.xlabel(r'$\beta$')
    plt.ylabel('energy')
    plt.savefig('annealing.png', dpi=800, bbox_inches='tight')
    plt.close()

    # plot num accepts and num rejects
    plt.plot(np.arange(num_steps) / 1e+3, run.num_accepts / 1e+3)
    plt.grid()
    plt.xlabel('metropolis step / $10^3$')
    plt.ylabel('number of acceptances / $10^3$')
    plt.savefig('num_accepts.png', dpi=800, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    mpl.use('Agg')

    # run example without a callback
    example(None)

    # run example with a callback that writes to a file
    with open('run.csv', 'w') as file:
        example(lambda string: print(string, file=file))
