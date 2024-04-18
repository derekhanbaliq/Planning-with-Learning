import numpy as np


class WaypointLoader:
    def __init__(self, map_name, csv_data=None):
        self.unit_dist = 0.2

        if (map_name == 'Spielberg' or map_name == 'MoscowRaceway' or map_name == 'Catalunya'
             ):
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3]  # coordinate matters!
            self.γ = csv_data[:, 4]
        elif (map_name == 'example' or map_name == 'icra' or map_name == 'levine' 
                or map_name == 'levine_2nd' or map_name == 'skir'):
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3] + np.pi / 2  # coordinate matters!
            # for i in range(self.θ.shape[0]):
            #     if self.θ[i] >= np.pi:
            #         self.θ[i] -= 2*np.pi
            #     elif self.θ[i] <= -np.pi:
            #         self.θ[i] += 2*np.pi
            
            self.γ = csv_data[:, 4]

