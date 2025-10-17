import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import os
def visualize_matrix(matrix, color_0=(1, 1, 1), color_1=(0, 0, 0), edge_color=(0.5, 0.5, 0.5), 
                     edge_width=1, font_size=12, save_path=True):
    """
    Function to visualize a binary (0–1) matrix. Supports RGB color definitions and allows adjusting grid and axis display.
    
    Parameters:
        - matrix: numpy.ndarray —— a 2D matrix containing values 0 and 1.
        - color_0: tuple —— RGB color for entries equal to 0; default is white (1, 1, 1).
        - color_1: tuple —— RGB color for entries equal to 1; default is black (0, 0, 0).
        - edge_color: tuple —— RGB color for the grid lines; default is gray (0.5, 0.5, 0.5).
        - edge_width: int —— width of the grid lines and the outer border.
        - font_size: int —— font size for the index numbers.
        - save_path: str —— file path where the image will be saved (including file name and extension), e.g. 'output/matrix_plot.png'.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("The input matrix must be two-dimensional numpy.ndarray")
    
    # Create a custom color mapping
    cmap = ListedColormap([color_0, color_1])
    
    # Plotting the Matrix
    plt.figure(figsize=(matrix.shape[1] * 0.7, matrix.shape[0] * 0.7))
    plt.imshow(matrix, cmap=cmap, interpolation="none")

    # Adding Grid Lines
    plt.gca().set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    plt.gca().grid(which="minor", color=edge_color, linestyle='-', linewidth=edge_width, zorder=0)
    
    # Add outer border (adjust width slightly to match visual effect)
    rect = patches.Rectangle(
        (-0.5, -0.5),  # Lower left corner coordinates
        matrix.shape[1],  # Matrix width
        matrix.shape[0],  # Matrix Height
        linewidth=edge_width * 1.5,  # Slightly adjusted to make them visually match
        edgecolor=edge_color,
        facecolor='none',
        zorder=1  # Stay on top of grid lines
    )
    plt.gca().add_patch(rect)

    # Setting up the axes
    plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(matrix.shape[1]), fontsize=font_size)
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=np.arange(matrix.shape[0]), fontsize=font_size)

    # Place the x-axis coordinate at the top
    plt.tick_params(axis='x', labeltop=True, labelbottom=False)
    
    # Remove outer border
    plt.tick_params(which="minor", size=0)
    plt.tick_params(axis='both', which='major', length=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Adjust canvas scale
    plt.tight_layout()
    
    # Save Image
    if save_path:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)  # Create a folder
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
# Example Matrix
matrix = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])

# Calling a function
visualize_matrix(matrix, color_0=[211/255,235/255,248/255], color_1=[246/255,174/255,69/255], edge_color=[139/255,139/255,140/255])
