def SVD_for_RT(source, target):
"""
source -- N*3
target -- N*3
"""
    mean_s = np.mean(source, axis=0)
    mean_t = np.mean(target, axis=0)
    source_m = source - mean_s
    target_m = target - mean_t
    S = np.zeros((3,3))
    for i in range(source.shape[0]):
        S += target_m[i,:].reshape((3,1)).dot(source_m[i].reshape((1,3)))
    U, A, V = np.linalg.svd(S)
    R = U.dot(V.transpose())
    if np.linalg.det(R)<0:
        R = -1*R
    t = mean_t - R.dot(mean_s)
    a = target - R.dot(source.transpose()).transpose()-t
    return R, t
