def rotation_translation_error(transforms, transforms_gt):
"""
transforms -- 4*4
transforms_gt -- 4*4
"""
    temp = transforms.dot(np.linalg.inv(transforms_gt))
    rotation_error_deg = 57.3 * np.arccos(min([abs(np.trace(temp[:3, :3]) - 1) / 2, 1]))
    translation_error_m = np.linalg.norm(transforms[:3, 3] - transforms_gt[:3, 3])
    return rotation_error_deg, translation_error_m
