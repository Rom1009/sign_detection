import numpy as np
import math
import random

class Affine_Transformation:
    def __init__(self):
        '''
        Initializes an empty list `A_list` to store the affine transformation matrices.
        '''
        self.A_list = []

    def compute_matrix(self):
        '''
        Input: None
        Output: numpy array (3x3 matrix)
        
        Description:
        Computes the combined affine transformation matrix by multiplying all matrices
        stored in `A_list`. The identity matrix is used as the starting point for the matrix.
        '''
        M = np.eye(3)
        for A in self.A_list:
            M = A.dot(M)
        return M

    def flip_horizontal(self):
        '''
        Input: None
        Output: None
        
        Description:
        Applies a horizontal flip to the transformation matrix. Adds the corresponding
        matrix for horizontal flipping to `A_list`.
        '''
        A = np.eye(3)
        A[0,0] = -1
        A[1,1] = 1
        self.A_list.append(A)

    def flip_vertical(self):
        '''
        Input: None
        Output: None
        
        Description:
        Applies a vertical flip to the transformation matrix. Adds the corresponding
        matrix for vertical flipping to `A_list`.
        '''
        A = np.eye(3)
        A[0,0] = 1
        A[1,1] = -1
        self.A_list.append(A)

    def rotation_radian(self, r):
        '''
        Input: r (float): Rotation angle in radians
        Output: None
        
        Description:
        Applies a rotation by the specified angle `r` (in radians). Adds the corresponding
        rotation matrix to `A_list`.
        '''
        A = np.eye(3)
        A[0,0] = math.cos(r)
        A[1,1] = math.cos(r)
        A[0,1] = -math.sin(r)
        A[1,0] = math.sin(r)
        self.A_list.append(A)

    def random_rotation(self, min_degree, max_degree):
        '''
        Input: 
        - min_degree (float): Minimum rotation angle in degrees.
        - max_degree (float): Maximum rotation angle in degrees.
        Output: None
        
        Description:
        Applies a random rotation between `min_degree` and `max_degree` (converted to radians).
        Adds the corresponding rotation matrix to `A_list`.
        '''
        min_radian = min_degree * (2 * math.pi) / 360
        max_radian = max_degree * (2 * math.pi) / 360
        random_radian = random.uniform(min_radian, max_radian)
        self.rotation_radian(random_radian)

    def scaling(self, x, y=None):
        '''
        Input: 
        - x (float): Scaling factor for the x-axis.
        - y (float, optional): Scaling factor for the y-axis. If `y` is not provided, it defaults to `x`.
        Output: None
        
        Description:
        Applies scaling in both the x and y directions. Adds the corresponding scaling matrix to `A_list`.
        '''
        if y is None:
            y = x
        A = np.eye(3)
        A[0,0] = x
        A[1,1] = y
        self.A_list.append(A)

    def noise(self, magnitude):
        '''
        Input: magnitude (float): Maximum random shift for the translation.
        Output: None
        
        Description:
        Applies a random translation (noise) in both x and y directions with values drawn from a uniform
        distribution in the range [-magnitude, magnitude]. Adds the corresponding matrix to `A_list`.
        '''
        A = np.eye(3)
        A[0,2] = np.random.uniform(-magnitude, magnitude)
        A[1,2] = np.random.uniform(-magnitude, magnitude)
        self.A_list.append(A)

    def skew_x_radian(self, r):
        '''
        Input: r (float): Skew angle in radians.
        Output: None
        
        Description:
        Applies a skew transformation along the x-axis by the angle `r`. Adds the corresponding matrix to `A_list`.
        '''
        A = np.eye(3)
        A[0, 1] = math.tan(r)
        self.A_list.append(A)

    def skew_y_radian(self, r):
        '''
        Input: r (float): Skew angle in radians.
        Output: None
        
        Description:
        Applies a skew transformation along the y-axis by the angle `r`. Adds the corresponding matrix to `A_list`.
        '''
        A = np.eye(3)
        A[1, 0] = math.tan(r)
        self.A_list.append(A)

    def skew_x_degree(self, d):
        '''
        Input: d (float): Skew angle in degrees.
        Output: None
        
        Description:
        Converts the angle `d` from degrees to radians and applies skew transformation along the x-axis.
        Adds the corresponding matrix to `A_list`.
        '''
        r = d * (2 * math.pi) / 360
        self.skew_x_radian(r)

    def skew_y_degree(self, d):
        '''
        Input: d (float): Skew angle in degrees.
        Output: None
        
        Description:
        Converts the angle `d` from degrees to radians and applies skew transformation along the y-axis.
        Adds the corresponding matrix to `A_list`.
        '''
        r = d * (2 * math.pi) / 360
        self.skew_y_radian(r)

    def translation(self, x, y):
        '''
        Input: 
        - x (float): Translation along the x-axis.
        - y (float): Translation along the y-axis.
        Output: None
        
        Description:
        Applies a translation transformation with offsets `x` and `y`. Adds the corresponding matrix to `A_list`.
        '''
        A = np.eye(3)
        A[0,2] = x
        A[1,2] = y
        self.A_list.append(A)

    def transform(self, data):
        '''
        Input: 
        - data (numpy array): Input data (e.g., landmark points) in the shape (num_points, 2).
        Output: Transformed data (numpy array).
        
        Description:
        Applies the accumulated affine transformations to the input `data`. First reshapes the input,
        then applies the affine transformations, and finally reshapes the data back to its original shape.
        '''
        data = data.copy()
        shape = data.shape
        data = data.reshape((-1, 2))
        data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        M = self.compute_matrix()
        data = data @ M.T
        data = (data / data[:, 2][:, None])[:, :2]
        data = data.reshape(shape)
        return data