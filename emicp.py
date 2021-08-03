import numpy as np
from scipy.optimize import least_squares
from tools import rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix, show_pair_pc, rigid_transform, downsample, points_denoise
from sklearn.neighbors import NearestNeighbors
import math
import concurrent.futures
import open3d as o3d
import scipy.io as scio
from scipy.spatial.transform import Rotation

class robust_icp():
    def __init__(self):
        self.delta = 0.02
        self.rotm_axis = "y"

    def em_icp(self, source, target):
        source = self.downsample(source, voxel_size=0.06)
        target = self.downsample(target, voxel_size=0.06)

        tar_scale = np.max(np.linalg.norm(target - np.min(target, axis=0), axis=1))
        src_scale = np.max(np.linalg.norm(source - np.min(source, axis=0), axis=1))
        ratio = tar_scale / src_scale
        src = source * ratio
        s = ratio
        mean_src = np.mean(src, axis=0)
        mean_tar = np.mean(target, axis=0)
        t = - mean_src + mean_tar
        src = s*source+t
        #
        v = np.zeros(4)
        res_1 = least_squares(self.cost_function, v, args=(src, target))
        v = res_1.x
        error = res_1.cost
        R = self.generate_rotm(v[0])
        t = v[1:4] + R.dot(t)
        # show_pair_pc(source, target)
        #
        # R, t, s, error = self.iterative0(source, target)

        src = ((s*source).dot(R.transpose()) + t)
        # show_pair_pc(src, target)
        tar_scale = np.max(np.linalg.norm(target - np.min(target, axis=0), axis=1))
        src_scale = np.max(np.linalg.norm(src - np.min(src, axis=0), axis=1))
        ratio = tar_scale / src_scale
        src = src * ratio
        v = np.zeros(4)
        v[:1] = 1.0
        res_1 = least_squares(self.scale_funtion, v, args=(src, target))
        error = res_1.cost
        x = res_1.x
        s = s*ratio*x[:1]
        t = ratio*x[:1]*t + x[1:]

        # show_pair_pc(src, target)

        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t


        return T, error, s

    def iteratve_icp(self, source, target):
        mean_source = np.mean(source, axis=0)
        mean_target = np.mean(target, axis=0)
        src = source - mean_source
        tar = target - mean_target
        R, t, s, error = self.iterative(src, tar)
        t = t + mean_target - R.dot(s*mean_source)
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T, error, s

    def SVD_T(self, source, target, iter=40):

        mean_s = np.mean(source, axis=0)
        mean_t = np.mean(target, axis=0)
        source_m = source - mean_s
        target_m = target - mean_t

        s = np.mean(np.linalg.norm(source_m * target_m, axis=1) / np.linalg.norm(source_m * source_m, axis=1))
        v = 20
        s_max = 1 + 1 / (v + iter)
        s_min = 1 - 1 / (v + iter)
        if s > s_max:
            s = s_max
        elif s < s_min:
            s = s_min
        source_m[:, 1] = 0
        target_m[:, 1] = 0
        S = target_m.transpose().dot(s*source_m)
        U, A, V = np.linalg.svd(S)
        R = U.dot(V)

        if np.linalg.det(R)<0:
            R = -1*R
        t = mean_t - s*R.dot(mean_s)
        src = s * source.dot(R.transpose()) + t
        # show_pair_pc(src, target)

        return R, t, s

    def SVD_T_1(self, source, target, iter):
        mean_s = np.mean(source, axis=0)
        mean_t = np.mean(target, axis=0)
        source_m = source - mean_s
        target_m = target - mean_t

        s = self.estimate_scale(source_m, target_m, iter)
        source_m[:, 1] = 0
        target_m[:, 1] = 0
        S = target_m.transpose().dot(source_m.dot(np.diag(s)))
        U, A, V = np.linalg.svd(S)
        R = U.dot(V)

        if np.linalg.det(R)<0:
            R = -1*R
        t = mean_t - np.diag(s).dot(R.dot(mean_s))

        return R, t, s

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def estimate_scale(self, src, tar, iter):
        v = 20
        s_max = 1 + 1/(np.array([v, v, v]) + iter)
        s_min = 1 - 1/(np.array([v, v, v]) + iter)
        s = np.ones(3)
        for i in range(3):
            temp = np.zeros_like(src)
            temp[:, i] = src[:, i]
            ss = np.sum(np.sum(tar * temp, axis=1)) / np.sum(np.sum(temp * temp, axis=1))
            if ss > s_max[i]:
                ss = s_max[i]
            if ss < s_min[i]:
                ss = s_min[i]
            s[i] = ss
        return s

    def first_k(self, dist, k=0.95):
        idx=np.argsort(dist[:,0])
        length = int(dist.shape[0]*k)
        idx = idx[:length]
        return idx

    def estimate_length(self, pt):
        pt = pt - np.min(pt, axis=0)
        pt_scale = np.max(pt, axis=0)
        return pt_scale

    def iterative0(self, source, target, iteration=50):
        R = np.eye(3)
        t = np.zeros(3)
        s = 1
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        for iter in range(iteration):
            src = (s*source).dot(R.transpose()) + t
            # show_pair_pc(src, target)
            tar_scale = np.max(np.linalg.norm(target[:,[0,2]] - np.min(target[:,[0,2]], axis=0), axis=1))
            src_scale = np.max(np.linalg.norm(src[:,[0,2]] - np.min(src[:,[0,2]], axis=0), axis=1))
            ratio = tar_scale / src_scale
            src = src*ratio

            # show_pair_pc(src, target)
            distance1, indices = nbrs1.kneighbors(src)


            tar = target[indices[:, 0], :3]
            idx = self.first_k(distance1, 0.9)
            tar = tar[idx, :]
            src = src[idx, :]
            # show_pair_pc(src, tar)
            R_new, t_new, s_new = self.SVD_T(src, tar, iter)

            R = R_new.dot(R)
            t = s_new*R_new.dot(t) + t_new
            s = s*s_new*ratio

            U, A, V = np.linalg.svd(R)
            R = U.dot(V)
            if np.linalg.det(R) < 0:
                R = -1 * R
        error = self._distance(src, target, 1)
        return R, t, s, error

    def iterative(self, source, target,iter=50):
        R = np.eye(3)
        t = np.zeros(3)
        s = 1
        error = 100
        for iter in range(iter):
            src = ((s*source).dot(R.transpose()) + t)
            tar_scale = self.estimate_length(target)
            src_scale = self.estimate_length(src)
            ratio = tar_scale / src_scale
            src *= ratio

            nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
            distance1, indices = nbrs1.kneighbors(src)

            tar = target[indices[:, 0], :]
            R_new, t_new, s_new = self.SVD_T_1(src, tar, iter)

            R = R_new.dot(R)
            t = s_new*R_new.dot(t) + t_new
            s = s*s_new*ratio

            U, A, V = np.linalg.svd(R)
            R = U.dot(V)
            if np.linalg.det(R) < 0:
                R = -1 * R
        error = self._distance(src, target, 1)
        return R, t, s, error

    def _distance(self, source, target, k=0.9):
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(source)
        distance1, _ = nbrs1.kneighbors(target)
        errors = np.mean(distance1)

        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, _ = nbrs1.kneighbors(source)
        idx = self.first_k(distance1, k)
        errors += np.mean(distance1[idx, 0])
        return errors

    def cost_function(self, x, *args):
        source = args[0]
        target = args[1]
        R = self.generate_rotm(x[0])
        t = x[1:]

        src = source.dot(R.transpose()) + t

        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, indices = nbrs1.kneighbors(src)

        errors = self.huber_loss(distance1[:,0])

        return errors

    def registration(self, x, *args):
        source = args[0]
        target = args[1]
        R = eulerAnglesToRotationMatrix([x[0], x[1], x[2]])
        t = x[3:6]
        src = source.dot(R.transpose()) + t
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, indices = nbrs1.kneighbors(src)
        errors = self.huber_loss(distance1[:,0])
        return errors

    def scale_funtion(self, x, *args):
        source = args[0]
        target = args[1]
        t = x[1:]
        s = x[:1]
        src = s*source + t
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(src)
        distance1, indices = nbrs1.kneighbors(target)
        errors = self.huber_loss(distance1[:, 0])

        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, indices = nbrs1.kneighbors(src)
        distance1 = self.huber_loss(distance1, 0.15)
        errors = np.concatenate((distance1[:, 0], errors), axis=0)

        return errors

    def huber_loss(self, errors, delta=0.08):
        # delta = 0.08
        temp = errors
        errors[temp <= delta] = np.sqrt(0.5) * errors[temp <= delta]
        errors[temp > delta] = np.sqrt(delta * errors[temp > delta] - 0.5 * delta * delta)
        return errors

    def generate_rotm(self, rot_deg):
        if self.rotm_axis == "x":
            rotm = Rotation.from_euler('x', rot_deg, degrees=True).as_dcm()
        elif self.rotm_axis == "y":
            rotm = Rotation.from_euler('y', rot_deg, degrees=True).as_dcm()
        else:
            rotm = Rotation.from_euler('z', rot_deg, degrees=True).as_dcm()
        return rotm

class global_em_icp(robust_icp):

    def __init__(self, steps, flag=0, rotm_axis="y"):
        """input
         --steps: divide the axis into how many bins
         --rotm_axis: the rotation ia along which axis
         output
         --trans: transformation from the source to target
         --error: registration error
        """
        gaps = 2*math.pi/steps*57.3
        self.rot_deg_list = [gaps*i + gaps/2  for i in range(steps)]
        self.rotm_axis = rotm_axis
        self.flag = flag
        # if flag == 1:
            # self.num_for_registration = 200
        # else:
        self.num_for_registration = 800

    def _resample(self, points):
        k = self.num_for_registration
        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]

    def downsample(self, ptc, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptc[:, :3])
        if ptc.shape[1] == 6:
            pcd.normals = o3d.utility.Vector3dVector(ptc[:, 3:])
            downpcd = pcd.voxel_down_sample(voxel_size)
            ptc = np.array(downpcd.points)
            normals = np.array(downpcd.normals)
            return np.concatenate((ptc, normals), axis=1)
        else:
            downpcd = o3d.voxel_down_sample(pcd, voxel_size)
            ptc = np.array(downpcd.points)
            return ptc

    def run(self, source, target):


        self.source_down = source
        self.target_down = target
        # self.t
        # self.tar_scale =  np.max(np.linalg.norm(self.target_down -np.min(self.target_down, axis=0), axis=1))
        # self.source = self.source
        # self.target  = self.target

        dist_error = []
        T_set = []
        s_set = []

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)
        for rot_deg, result in zip(self.rot_deg_list, executor.map(self.multi_process, self.rot_deg_list)):
            T_set.append(result[0])
            dist_error.append(result[1])
            s_set.append(result[2])
        executor.shutdown(wait=True)
        #
        # for i in range(len(self.rot_deg_list)):
        #     result = self.multi_process(self.rot_deg_list[i])
        #     T_set.append(result[0])
        #     dist_error.append(result[1])
        #     s_set.append(result[2])
            # trans = result[0]
            # print(result[1])
            # show_pair_pc(rigid_transform(self.source_down, trans), self.target_down)

        idx = np.argmin(dist_error)
        trans = T_set[idx]
        error = dist_error[idx]
        tar_scale = np.max(np.linalg.norm(self.target_down - np.min(self.target_down, axis=0), axis=1))  # [:,[0,2]]
        error /= tar_scale
        # src_trans = rigid_transform(self.source_down, trans)
        # R, t, s, error = self.iterative(src_trans, self.target_down, iter=500)
        # show_pair_pc(rigid_transform(self.source_down,trans), self.target_down)
        # trans[:3,:3] = s*R.dot(trans[:3,:3])
        # trans[:3,3] = s*R.dot(trans[:3,3]) + t
        # s = s_set[idx]



        return trans, error

    def multi_process(self, rot_deg):

        tar_mean = np.max(self.target_down, axis=0)
        tar = self.target_down - tar_mean
        tar_scale = np.max(np.linalg.norm(tar[:,[0,2]] -np.min(tar[:,[0,2]], axis=0), axis=1)) #[:,[0,2]]

        rotm = self.generate_rotm(rot_deg)
        src = self.source_down.dot(rotm.transpose())
        src_mean = np.max(src, axis=0)
        src = src - src_mean
        src_scale = np.max(np.linalg.norm(src[:,[0,2]]  - np.min(src[:,[0,2]], axis=0), axis=1))
        ratio = tar_scale / src_scale
        src = src * ratio
        # show_pair_pc(src, tar)
        trans, error, s = self.em_icp(src, tar)
        s *= ratio
        trans[:3,  3] = -(trans[:3, :3]*s).dot(src_mean) + trans[:3, 3] + tar_mean
        trans[:3, :3] = (trans[:3, :3]*s).dot(rotm)
        return (trans, error, s)



if __name__=="__main__":
    source = np.random.random((1024, 3))
    R = eulerAnglesToRotationMatrix([0, 3.14/2, 0])
    target = source.dot(R.transpose())
    em_icp = global_em_icp(12)
    trans, error = em_icp.run(source, target)
    show_pair_pc(target, rigid_transform(source, trans) + np.array([0,0.1,1]))

