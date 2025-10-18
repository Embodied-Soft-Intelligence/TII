import os
import torch
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
import datasets.transforms as T
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder_1, label_folder_2, label_folder_3, transform=None):
        if not os.path.exists(image_folder):
            raise ValueError(f"Image folder {image_folder} does not exist.")
        if not os.path.exists(label_folder_1):
            raise ValueError(f"Label folder {label_folder_1} does not exist.")
        if not os.path.exists(label_folder_2):
            raise ValueError(f"Label folder {label_folder_2} does not exist.")
        if not os.path.exists(label_folder_3):
            raise ValueError(f"Label folder {label_folder_3} does not exist.")
        
        self.image_folder = image_folder
        self.label_folder_1 = label_folder_1
        self.label_folder_2 = label_folder_2
        self.label_folder_3 = label_folder_3
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
        
        if not self.image_files:
            raise ValueError(f"No image files found in {image_folder}.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        label_path_1 = os.path.join(self.label_folder_1, os.path.splitext(image_file)[0] + '.txt')
        label_path_2 = os.path.join(self.label_folder_2, os.path.splitext(image_file)[0] + '.txt')
        label_path_3 = os.path.join(self.label_folder_3, os.path.splitext(image_file)[0] + '.txt')

        classes = []
        boxes = []
        size = []

        try:
            with open(label_path_1, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    try:
                        if len(parts) < 7:
                            raise ValueError(f"Insufficient data in line: {line.strip()}")
                        class_id = int(parts[-7])
                        box_coords = list(map(float, parts[-6:-2]))
                        size = [int(parts[-1]), int(parts[-2])]
                        classes.append(class_id)
                        boxes.append(box_coords)
                    except ValueError as e:
                        print(f"Error parsing line in {label_path_1}: {line.strip()} - {e}")
                        continue
        except FileNotFoundError as e:
            print(f"Label file {label_path_1} not found: {e}")
        
        classes = torch.tensor(classes, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        size = torch.tensor(size, dtype=torch.int64)

        adj_matrix = []
        try:
            with open(label_path_2, 'r') as f:
                for line in f:
                    try:
                        row = list(map(int, line.strip().split()))
                        adj_matrix.append(row)
                    except ValueError as e:
                        print(f"Error parsing line in {label_path_2}: {line.strip()} - {e}")
                        continue
        except FileNotFoundError as e:
            print(f"Label file {label_path_2} not found: {e}")
        
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.int32)

        grasp_points = []
        grasp_difficulty = []
        grasp_widths = []
        grasp_classes = []
        grasp_angles = []
        grasp_heights = []
        try:
            with open(label_path_3, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    try:
                        if len(parts) < 8:
                            raise ValueError(f"Insufficient data in line: {line.strip()}")
                        grasp_coords = list(map(float, parts[:8]))
                        points = [(grasp_coords[i], grasp_coords[i+1]) for i in range(0, len(grasp_coords), 2)]

                        centerX = sum([p[0] for p in points]) / 4
                        centerY = sum([p[1] for p in points]) / 4

                        midX = (points[1][0] + points[2][0]) / 2
                        midY = (points[1][1] + points[2][1]) / 2

                        angle = self.angle_between_points(centerX, centerY, midX, midY)

                        grasp_id = int(parts[-1])
                        difficulty_char = parts[-3]
                        difficulty = self.map_difficulty(difficulty_char)

                        grasp_points.append(points)
                        grasp_angles.append(angle)
                        grasp_classes.append(grasp_id)
                        grasp_difficulty.append(difficulty)

                    except ValueError as e:
                        print(f"Error parsing line in {label_path_3}: {line.strip()} - {e}")
                        continue
        except FileNotFoundError as e:
            print(f"Label file {label_path_3} not found: {e}")

        grasp_points = torch.tensor(grasp_points, dtype=torch.float32)
        grasp_angles = torch.tensor(grasp_angles, dtype=torch.float32)
        grasp_classes = torch.tensor(grasp_classes, dtype=torch.int64)
        grasp_difficulties = torch.tensor(grasp_difficulty, dtype=torch.int64)
        grasp_widths = torch.tensor(grasp_widths, dtype=torch.float32)
        grasp_heights = torch.tensor(grasp_heights, dtype=torch.float32)
        image_id = os.path.splitext(image_file)[0]
        image_id_numeric = torch.tensor(int(image_id))

        labels = {
            'image_id': image_id_numeric,
            'labels': classes,
            'boxes': boxes,
            'adj': adj_matrix,
            'grasp_points': grasp_points,
            'grasp_angles': grasp_angles,
            'grasp_classes': grasp_classes,
            'grasp_difficulties': grasp_difficulties,
            'grasp_widths': grasp_widths,
            'grasp_heights': grasp_heights,
            'orig_size': size,
            'size': size
        }

        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image, labels = self.transform(image, labels)
        grasp_angles = labels['grasp_angles']
        grasp_angles_cls = ((grasp_angles + 89.99) // 10).long()
        labels['grasp_angles_cls'] = grasp_angles_cls

        return image, labels
    
    def map_difficulty(self, char):
        mapping = {'e': 1, 'h': 2}
        return mapping.get(char, 0)
    
    def angle_between_points(self, x1, y1, x2, y2):

        delta_x = x2 - x1
        delta_y = y2 - y1

        angle = math.degrees(math.atan2(delta_y, delta_x))

        if angle * 2 <= -180:
            angle = (2 * angle + 360) / 2
        elif angle * 2 > 180:
            angle = (2 * angle - 360) / 2
        else:
            angle = angle

        return angle
    
    def calculate_euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def make_vmrd_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResize(scales, max_size=1333),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set):

    PATHS = {
        "train": ('/data/VMRD/images/train', '/data/VMRD/labels/train','/data/VMRD/adj/train', '/data/VMRD/grasps/train'),
        "val":  ('/data/VMRD/images/val', '/data/VMRD/labels/val','/data/VMRD/adj/val', '/data/VMRD/grasps/val'),
    }

    img_folder, label_folder, adj_folder, grasp_folder = PATHS[image_set]
    dataset= CustomDataset(img_folder , label_folder, adj_folder, grasp_folder, transform=make_vmrd_transforms(image_set))
    return dataset


def wh_calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def visualize_grasp_points(image, labels):
    draw = ImageDraw.Draw(image)
    
    grasp_points = labels.get('grasp_points', [])
    grasp_angles = labels.get('grasp_angles', [])

    for i, (points, angle) in enumerate(zip(grasp_points, grasp_angles)):

        center_x = sum([p[0] for p in points]) / 4
        center_y = sum([p[1] for p in points]) / 4

        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        width = wh_calculate_euclidean_distance(points[0][0], points[0][1], points[1][0], points[1][1])
        height = wh_calculate_euclidean_distance(points[1][0], points[1][1], points[2][0], points[2][1])

        half_width = width / 2
        half_height = height / 2

        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        rotated_corners = [
            (
                center_x + x * cos_angle - y * sin_angle,
                center_y + x * sin_angle + y * cos_angle
            )
            for x, y in corners
        ]

        rotated_corners = [(int(x), int(y)) for x, y in rotated_corners]
        draw.polygon(rotated_corners, outline='green', width=2)

        try:
            draw.ellipse((center_x - 3, center_y - 3, center_x + 3, center_y + 3), fill='red', outline='red')
        except Exception as e:
            print(f"Error drawing ellipse at center: {e}")

    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()