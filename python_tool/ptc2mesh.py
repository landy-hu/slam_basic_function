import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage import measure
from tools.tools import show_pair_pc
class ptc2mesh():
    def __init__(self):
        self.voxel_size = 0.5
    def volumn_size(self, ptc):
        self.xyz_max = np.max(ptc, axis=0)
        self.xyz_min = np.min(ptc, axis=0)
        self.voxel_shape =  ((self.xyz_max- self.xyz_min)/self.voxel_size).astype(np.int)

    def meshwrite(self, filename, verts, faces, norms, colors):
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (verts.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(verts.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                verts[i, 0], verts[i, 1], verts[i, 2], norms[i, 0], norms[i, 1], norms[i, 2], colors[i, 0],
                colors[i, 1],
                colors[i, 2]))

        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

        ply_file.close()
    def smooth(self, likelyhood_value):
        x_1 = likelyhood_value[1:-1, 1: -1, 1: -1] + likelyhood_value[1: -1, 1: -1, 0:-2] + likelyhood_value[1: -1,0:-2,1: -1] + likelyhood_value[0:-2, 1: -1, 1: -1]
        x_2 = likelyhood_value[0:-2, 0:-2, 0:-2] + likelyhood_value[0:-2, 0:-2, 1:-1] + likelyhood_value[0:-2, 1:-1,0:-2] + likelyhood_value[1:-1,0:-2, 0:-2]
        likelyhood_value[1: -1, 1: -1, 1: -1] = (x_1 + x_2) / 8
        return likelyhood_value
    def run(self, pcd):
        self.voxel_size = 0.03
        self.volumn_size(pcd)
        yv, xv, zv = np.meshgrid(
            np.linspace(self.xyz_min[1] + self.voxel_size / 2, self.xyz_max[1] - self.voxel_size / 2,
                        self.voxel_shape[1]),
            np.linspace(self.xyz_min[0] + self.voxel_size / 2, self.xyz_max[0] - self.voxel_size / 2,
                        self.voxel_shape[0]),
            np.linspace(self.xyz_min[2] + self.voxel_size / 2, self.xyz_max[2] - self.voxel_size / 2,
                        self.voxel_shape[2])
        )
        xv = np.asarray(xv.flat).reshape(-1, 1)
        yv = np.asarray(yv.flat).reshape(-1, 1)
        zv = np.asarray(zv.flat).reshape(-1, 1)
        grid = np.concatenate([xv, yv, zv], axis=1)


        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pcd)
        dist, _ = knn.kneighbors(X=grid, return_distance=True)
        df = dist.reshape(self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2])
        df[df > 0.5*self.voxel_size] = 1
        df[df <=  0.5*self.voxel_size] = 0
        df = df - 1
        df[df == -1] = 1
        df = df - 0.5
        df = self.smooth(df)
        df = self.smooth(df)

        verts, faces, normals, values = measure.marching_cubes_lewiner(df, level=0)
        verts = (verts/self.voxel_shape ) *(self.xyz_max - self.xyz_min) + self.xyz_min
        show_pair_pc(pcd, verts)
        # colors = np.array([191,39,35]).reshape(1,-1)*np.ones_like(verts)
        # name = './test.ply'
        # self.meshwrite(name, verts, faces, normals, colors)

        # tdf = 1.0 - np.where(df / (self.voxel_size * 0.9) < 1.0, 0, 1.0)
        return verts, faces, normals