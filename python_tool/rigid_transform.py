def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
     xyz -- N*3
     transformation -- 4*4
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]
