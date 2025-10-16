import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import os
def visualize_matrix(matrix, color_0=(1, 1, 1), color_1=(0, 0, 0), edge_color=(0.5, 0.5, 0.5), 
                     edge_width=1, font_size=12, save_path=True):
    """
    可视化0-1矩阵的函数，支持RGB颜色定义，并调整网格和坐标显示。
    
    参数:
        - matrix: numpy.ndarray, 包含0和1的二维矩阵
        - color_0: tuple, 表示0对应的RGB颜色，默认白色 (1, 1, 1)
        - color_1: tuple, 表示1对应的RGB颜色，默认黑色 (0, 0, 0)
        - edge_color: tuple, 方格网格线的RGB颜色，默认灰色 (0.5, 0.5, 0.5)
        - edge_width: int, 网格线和外围边框的宽度
        - font_size: int, 索引数字的字体大小
        - save_path: str, 保存图片的文件路径（包含文件名和扩展名，例如 'output/matrix_plot.png'）
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("输入的矩阵必须是二维的 numpy.ndarray")
    
    # 创建自定义颜色映射
    cmap = ListedColormap([color_0, color_1])
    
    # 绘制矩阵
    plt.figure(figsize=(matrix.shape[1] * 0.7, matrix.shape[0] * 0.7))
    plt.imshow(matrix, cmap=cmap, interpolation="none")

    # 添加网格线
    plt.gca().set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    plt.gca().grid(which="minor", color=edge_color, linestyle='-', linewidth=edge_width, zorder=0)
    
    # 添加外围边框（略微调整宽度以匹配视觉效果）
    rect = patches.Rectangle(
        (-0.5, -0.5),  # 左下角坐标
        matrix.shape[1],  # 矩阵宽度
        matrix.shape[0],  # 矩阵高度
        linewidth=edge_width * 1.5,  # 稍微调整，使视觉上匹配
        edgecolor=edge_color,
        facecolor='none',
        zorder=1  # 确保在网格线上方绘制
    )
    plt.gca().add_patch(rect)

    # 设置坐标轴
    plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(matrix.shape[1]), fontsize=font_size)
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=np.arange(matrix.shape[0]), fontsize=font_size)

    # 将 x 轴坐标放到上方
    plt.tick_params(axis='x', labeltop=True, labelbottom=False)
    
    # 移除外部边框
    plt.tick_params(which="minor", size=0)
    plt.tick_params(axis='both', which='major', length=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # 调整画布比例
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)  # 创建文件夹
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
# 示例矩阵
matrix = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
])

# 调用函数
visualize_matrix(matrix, color_0=[211/255,235/255,248/255], color_1=[246/255,174/255,69/255], edge_color=[139/255,139/255,140/255])
