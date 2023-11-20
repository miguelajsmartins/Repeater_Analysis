import matplotlib.pyplot as plt

#define style of axis
def set_style(ax, title, title_x, title_y, size_font):

    ax.set_title(title)
    ax.set_xlabel(title_x, fontsize=size_font)
    ax.set_ylabel(title_y, fontsize=size_font)
    ax.tick_params(axis='both', labelsize=size_font)

    return ax

#define a function to define the style of a color bar
def set_cb_style(cb, title, limits, label_size):

    cb.ax.set_ylabel(title, fontsize=label_size)
    cb.ax.set_ylim(limits[0], limits[1])
    cb.ax.tick_params(labelsize=label_size)

    return cb
