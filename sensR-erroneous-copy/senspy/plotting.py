import matplotlib.pyplot as plt

__all__ = ["plot_psychometric"]

def plot_psychometric(dprimes, pcs):
    """Simple plotting of psychometric function."""
    fig, ax = plt.subplots()
    ax.plot(dprimes, pcs, marker="o")
    ax.set_xlabel("d-prime")
    ax.set_ylabel("Proportion correct")
    return fig, ax
