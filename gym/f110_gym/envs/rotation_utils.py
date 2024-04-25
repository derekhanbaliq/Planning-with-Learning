"""
    helper functions for the lidar scan
    Author: Derek Zhou, Biao Wang, Tian Tan
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.path as mpath

class RotationUtil():
    def __init__(self, map_resolution) -> None:
        self.map_resolution = map_resolution
        
        self.width_car = 0.22
        self.length_car = 0.32
        
        self.diag_car = np.sqrt(self.width_car**2 + self.length_car**2)
        
        self.half_width = np.round(self.width_car / 2 / self.map_resolution).astype(int)
        self.half_length = np.round(self.length_car / 2 / self.map_resolution).astype(int)
        
        self.car_alpha = np.arctan2(self.half_width, self.half_length)
        
        # float half diag line 
        self.half_diag = self.diag_car / 2 / self.map_resolution
        # int half diag line
        self.half_diag_int = np.ceil(self.half_diag).astype(int)
    
    def sort_vertices_clockwise(self, vertices):
            centroid = np.mean(vertices, axis=0)
            def sort_key(vertex):
                rel_vector = vertex - centroid
                angle = math.atan2(rel_vector[1], rel_vector[0])
                return angle
            sorted_vertices = sorted(vertices, key=sort_key)
            return np.array(sorted_vertices)


    def find_points_inside(self, origin_x_index, origin_y_index, theta):
        dy1 = np.round(self.half_diag * np.cos(theta - self.car_alpha)).astype(int)
        dx1 = np.round(self.half_diag * np.sin(theta - self.car_alpha)).astype(int)
        
        dy2 = np.round(self.half_diag * np.cos(theta + self.car_alpha)).astype(int)
        dx2 = np.round(self.half_diag * np.sin(theta + self.car_alpha)).astype(int)
        
        # dx1 += 1 if dx1 > 0 else -1
        # dx2 += 1 if dx2 > 0 else -1
        # dy1 += 1 if dy1 > 0 else -1
        # dy2 += 1 if dy2 > 0 else -1
        
        vertices = np.array([
                            [origin_y_index + dy1, origin_x_index + dx1], 
                            [origin_y_index + dy2, origin_x_index + dx2],
                            [origin_y_index - dy1, origin_x_index - dx1], 
                            [origin_y_index - dy2, origin_x_index - dx2]
                            ])
        
        sorted_vertices = self.sort_vertices_clockwise(vertices)
        first_point = sorted_vertices[0,:]
        sorted_vertices = np.vstack([sorted_vertices, first_point])
        
        
        rows = np.arange(origin_y_index-2*self.half_diag_int, origin_y_index+2*self.half_diag_int, 1)
        cols = np.arange(origin_x_index-2*self.half_diag_int, origin_x_index+2*self.half_diag_int, 1)
        row_indices, col_indices = np.meshgrid(rows, cols, indexing='ij')
        points = np.vstack([row_indices.ravel(), col_indices.ravel()]).T
        path = mpath.Path(sorted_vertices)

        # Use the path to check which grid points are within the rectangle
        inside_index = path.contains_points(points, radius=0)
        inside_points = points[inside_index]
        
        return inside_points, points, sorted_vertices