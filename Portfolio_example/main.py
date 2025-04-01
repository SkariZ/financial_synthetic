"""
Run through the main.py file to execute the portfolio optimization and simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import naive_portfolio as npf
import monkey_portfolio as mp
import minimum_variance_portfolio as mvp2
import mean_variance_portfolio as mvp
import sharpe_ratio_portfolio as srp



# Run their scripts

if __name__ == "__main__":
    # Run the naive portfolio example
    print("Naive Portfolio Example")
    print("===================================")
    npf.main()

    # Run the monkey portfolio example
    print("Monkey Portfolio Example")
    print("===================================")
    mp.main()

    # Run the minimum variance portfolio example
    print("Minimum Variance Portfolio Example")
    print("===================================")
    mvp2.main()

    # Run the mean variance portfolio example
    print("Mean Variance Portfolio Example")
    print("===================================")
    mvp.main()

    # Run the sharpe ratio portfolio example
    print("Sharpe Ratio Portfolio Example")
    print("===================================")
    srp.main()