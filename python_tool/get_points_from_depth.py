def get_points_from_depth(depth_im, color_im, cam_intr):
    valid_pts = np.logical_and(depth_im > 0, depth_im < 5)
    pix_x,  pix_y = np.where(valid_pts)
    pix_z = depth_im[pix_x, pix_y]
    color = color_im[pix_x, pix_y, :]/255.0
    pix_y = (pix_y - cam_intr[0, 2]) * pix_z / cam_intr[0, 0]
    pix_x = (pix_x - cam_intr[1, 2]) * pix_z / cam_intr[1, 1]
    pixyz = np.concatenate((np.expand_dims(pix_y, axis=1), np.expand_dims(pix_x, axis=1), np.expand_dims(pix_z, axis=1)), axis=1)
    return pixyz, color
