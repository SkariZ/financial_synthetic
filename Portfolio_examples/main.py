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
    w_npf, p_npf = npf.main()

    # Run the monkey portfolio example
    print("Monkey Portfolio Example")
    print("===================================")
    w_mp, p_mp = mp.main()

    # Run the minimum variance portfolio example
    print("Minimum Variance Portfolio Example")
    print("===================================")
    w_mpv2, p_mpw2 = mvp2.main()

    # Run the mean variance portfolio example
    print("Mean Variance Portfolio Example")
    print("===================================")
    w_mvp, p_mvp = mvp.main()

    # Run the sharpe ratio portfolio example
    print("Sharpe Ratio Portfolio Example")
    print("===================================")
    w_srp, p_srp = srp.main()

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(p_npf, label='Naive Portfolio', linewidth=2)
    plt.plot(p_mp, label='Monkey Portfolio', linewidth=2)
    plt.plot(p_mpw2, label='Minimum Variance Portfolio', linewidth=2)
    plt.plot(p_mvp, label='Mean Variance Portfolio', linewidth=2)
    plt.plot(p_srp, label='Sharpe Ratio Portfolio', linewidth=2)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time (Days)')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig('../Imgs/Portfolio_Comparison.png', bbox_inches='tight', dpi=300)
    plt.show()
