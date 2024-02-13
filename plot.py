from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from random import choice

# Plot settings
figsize = (6,6)
rect_color = 'lightblue'
edge_color = 'black'
alpha = 0.5

def rotate_items(genome, items):
    new_items = []
    for item, gene in zip(items, genome):
        if gene[2] == 0:
            new_items.append(item)
        elif gene[2] == 1:
            new_items.append((item[1], item[0]))
    return new_items

def rotate_items_3d(genome, items):
    new_items = []
    for item, gene in zip(items, genome):
        w,h,d = item[0], item[1], item[2]

        if gene[3] == 1:
            temp = h
            h = d
            d = temp
        if gene[4] == 1:
            temp = w
            w = d
            d = temp
        if gene[5] == 1:
            temp = w
            w = h
            h = temp

        new_items.append((w,h,d))
    return new_items

def plot_configuration(configuration, items, W, H):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set the ticks for the main plot
    ax.set_xlim([0,W])
    ax.set_ylim([0,H])
    ax.set_xticks(range(0, int(W)+1, 5))
    ax.set_yticks(range(0, int(H)+1, 5))

    # Draw the placed rects in main plot
    for position, item in zip(configuration, items):
        box = Rectangle(position, item[0], item[1], fc=rect_color, ec=edge_color, alpha=alpha)
        ax.add_patch(box)

    plt.title('Best Configuration')
    plt.show()

def plot_configuration_with_rotation(configuration, items, W, H):
    dimensions = [(conf[0], conf[1]) for conf in configuration]
    new_items = rotate_items(configuration, items)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Set the ticks for the main plot
    ax.set_xlim([0,W])
    ax.set_ylim([0,H])
    ax.set_xticks(range(0, W+1, 5))
    ax.set_yticks(range(0, H+1, 5))

    # Draw the placed rects in main plot
    for position, item in zip(dimensions, new_items):
        box = Rectangle(position, item[0], item[1], fc=rect_color, ec=edge_color, alpha=alpha)
        ax.add_patch(box)

    plt.title('Best Configuration')
    plt.show()


def plot_fitness(best_fit, worst_fit, n_gen, fit_lim):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(best_fit, label='Best fitness')
    ax.plot(worst_fit, label='Worst fitness')
    ax.hlines(y=fit_lim, xmin=0, xmax=n_gen, colors='black', label='Optimal', linestyles='--')
    ax.set_xlim([0, n_gen])
    #ax.set_ylim([0, fit_lim])
    ax.legend()
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.show()


def plot_boxes(ax, boxes_pos, boxes_dim, multi_color):
    for pos, dim in zip(boxes_pos, boxes_dim):
        if multi_color:
            color = "#"+''.join([choice('0123456789ABCDEF') for j in range(6)])
        else:
            color = 'lightblue'
        x, y, z = pos
        dx, dy, dz = dim
        ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color, edgecolors=edge_color, alpha=alpha)
        
## Plot configuration
def plot_configuration_3d(configuration, items, W, H, D, multi_color=False):

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot limits
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_zlim([0, D])

    # Plot the boxes
    plot_boxes(ax, configuration, items, multi_color=multi_color)

    # Adjust aspect ratio to make the plot rectangular
    ax.set_box_aspect([W, H, D])

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Bin Packing Solution')

    # Show the plot
    plt.show()

def plot_configuration_3d_with_rotation(configuration, items, W, H, D, multi_color=False):
    dimensions = [(conf[0], conf[1], conf[2]) for conf in configuration]
    new_items = rotate_items_3d(configuration, items)

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot limits
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_zlim([0, D])

    # Plot the boxes
    plot_boxes(ax, dimensions, new_items, multi_color=multi_color)

    # Adjust aspect ratio to make the plot rectangular
    ax.set_box_aspect([W, H, D])

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Bin Packing Solution')

    # Show the plot
    plt.show()


def plot_boxes_interactive(fig, boxes_pos, boxes_dim):
    for pos, dim in zip(boxes_pos, boxes_dim):
        x, y, z = pos
        dx, dy, dz = dim
        #colors = ["#"+''.join([choice('0123456789ABCDEF') for j in range(6)]) for i in range(2)]
        color = "#"+''.join([choice('0123456789ABCDEF') for j in range(6)])
        
        fig.add_trace(go.Mesh3d(x=[x, x, x+dx, x+dx, x, x, x+dx, x+dx],
                                y=[y, y+dy, y+dy, y, y, y+dy, y+dy, y],
                                z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
                                #colorscale=[[0, colors[0]],[1, colors[1]]],
                                # Intensity of each vertex, which will be interpolated and color-coded
                                #intensity = np.linspace(0, 1, 8, endpoint=True),
                                # i, j and k give the vertices of triangles
                                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                                name='y',
                                showscale=False,
                                opacity=1.0,
                                color=color,
                                flatshading=True                         
                                ))
        
def plot_configuration_3d_interactive(configuration, items, W, H, D):

    # Create the interactive 3D plot
    fig = go.Figure()

    # Plot the boxes
    plot_boxes_interactive(fig, configuration, items)

    # Set layout
    fig.update_layout(scene=dict(aspectmode="data",
                                 xaxis=dict(range=[0, W]),
                                 yaxis=dict(range=[0, H]),
                                 zaxis=dict(range=[0, D]),
                                 aspectratio=dict(x=1, y=1, z=1)))

    # Show the interactive plot
    fig.show()

def plot_configuration_3d_interactive_with_rotation(configuration, items, W, H, D):
    dimensions = [(conf[0], conf[1], conf[2]) for conf in configuration]
    new_items = rotate_items_3d(configuration, items)

    # Create the interactive 3D plot
    fig = go.Figure()

    # Plot the boxes
    plot_boxes_interactive(fig, dimensions, new_items)

    # Set layout
    fig.update_layout(scene=dict(aspectmode="data",
                                 xaxis=dict(range=[0, W]),
                                 yaxis=dict(range=[0, H]),
                                 zaxis=dict(range=[0, D]),
                                 aspectratio=dict(x=W, y=H, z=D)))

    # Show the interactive plot
    fig.show()