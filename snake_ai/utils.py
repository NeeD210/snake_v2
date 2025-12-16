import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(losses, title="Training Loss"):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def one_hot_encode(state):
    # Just a placeholder if we need it
    pass
