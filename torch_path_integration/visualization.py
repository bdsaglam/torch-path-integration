import matplotlib.pyplot as plt


def plot_location_predictions(initial_location, prediction, target):
    batch_size = prediction.shape[0]
    fig, axes = plt.subplots(nrows=batch_size, ncols=1, figsize=(4, batch_size * 4))
    for i in range(batch_size):
        ax = axes[i] if batch_size > 1 else axes
        ax.scatter(initial_location[i, :, 0], initial_location[i, :, 1], c='black', marker='x')
        ax.plot(target[i, :, 0], target[i, :, 1], c='blue', marker='.')
        ax.plot(prediction[i, :, 0], prediction[i, :, 1], c='red', marker='.')
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.invert_yaxis()
    return fig
