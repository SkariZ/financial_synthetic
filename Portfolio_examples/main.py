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

# Run the main function for each portfolio example
if __name__ == "__main__":
    # Run the naive portfolio example
    print("Naive Portfolio Example")
    print("===================================")
    performance_npf, p_npf = npf.main()

    # Run the monkey portfolio example
    print("Monkey Portfolio Example")
    print("===================================")
    performance_mp, p_mp = mp.main()

    # Run the minimum variance portfolio example
    print("Minimum Variance Portfolio Example")
    print("===================================")
    performance_mpv2, p_mpw2 = mvp2.main()

    # Run the mean variance portfolio example
    print("Mean Variance Portfolio Example")
    print("===================================")
    performance_mvp, p_mvp = mvp.main()

    # Run the sharpe ratio portfolio example
    print("Sharpe Ratio Portfolio Example")
    print("===================================")
    performance_srp, p_srp = srp.main()

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(p_npf, label='Naive Portfolio', linewidth=2.5)
    plt.plot(p_mp, label='Monkey Portfolio', linewidth=2.5)
    plt.plot(p_mpw2, label='Minimum Variance Portfolio', linewidth=2.5)
    plt.plot(p_mvp, label='Mean Variance Portfolio', linewidth=2.5)
    plt.plot(p_srp, label='Sharpe Ratio Portfolio', linewidth=2.5)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time (Days)')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig('../Imgs/Portfolio_Comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Create a table with performance metrics for each portfolio
    performance_table = pd.DataFrame({
        'Portfolio': ['Naive', 'Monkey', 'Minimum Variance', 'Mean Variance', 'Sharpe Ratio'],
        'Final Portfolio Value': [
            performance_npf['Final Portfolio Value'],
            performance_mp['Final Portfolio Value'],
            performance_mpv2['Final Portfolio Value'],
            performance_mvp['Final Portfolio Value'],
            performance_srp['Final Portfolio Value']
        ],
        'Total Return (%)': [
            performance_npf['Total Return (%)'],
            performance_mp['Total Return (%)'],
            performance_mpv2['Total Return (%)'],
            performance_mvp['Total Return (%)'],
            performance_srp['Total Return (%)']
        ],
        'Maximum Drawdown (%)': [
            performance_npf['Maximum Drawdown (%)'],
            performance_mp['Maximum Drawdown (%)'],
            performance_mpv2['Maximum Drawdown (%)'],
            performance_mvp['Maximum Drawdown (%)'],
            performance_srp['Maximum Drawdown (%)']
        ],
        'Annualized Return (%)': [
            performance_npf['Annualized Return (%)'],
            performance_mp['Annualized Return (%)'],
            performance_mpv2['Annualized Return (%)'],
            performance_mvp['Annualized Return (%)'],
            performance_srp['Annualized Return (%)']
        ],
        'Sharpe Ratio': [
            performance_npf['Sharpe Ratio'],
            performance_mp['Sharpe Ratio'],
            performance_mpv2['Sharpe Ratio'],
            performance_mvp['Sharpe Ratio'],
            performance_srp['Sharpe Ratio']
        ]
    })
    performance_table.set_index('Portfolio', inplace=True)

    # Round the performance metrics to 2 decimal places
    performance_table = performance_table.round(3)

    # Do a nice plot of the performance metrics table
    fig, ax = plt.subplots(figsize=(10, 2))  # Set the size of the figure
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=performance_table.values,
                     colLabels=performance_table.columns,
                     rowLabels=performance_table.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)  # Activate auto font size
    table.set_fontsize(10)  # Set font size
    table.scale(1.2, 1.2)  # Scale the table to make it larger
    # Color the table
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_facecolor('#40466e')  # Header color
            cell.set_text_props(color='w')  # Header text color
        else:
            cell.set_facecolor('#f1f1f2')  # Body color

    # Color maximum and minimum values in each column with green and red
    for j in range(len(performance_table.columns)):
        col = performance_table.iloc[:, j]
        max_val = col.max()
        min_val = col.min()
        for i in range(len(col)):
            if col[i] == max_val:
                table[(i + 1, j)].set_facecolor('#00FF00')  # Green for maximum
            elif col[i] == min_val:
                table[(i + 1, j)].set_facecolor('#FF0000')  # Red for minimum
            
    plt.title('Portfolio Performance Metrics Comparison', fontsize=14)
    plt.savefig('../Imgs/Portfolio_Performance_Metrics.png', bbox_inches='tight', dpi=300)
    plt.show()







