function [pixyz, color] = get_points_from_depth(depth_im, color_im, cam_intr)
    % depth_im = depth_im/fl;
    depth_im(depth_im>5) = 0;
    depth_im(depth_im<0.5) = 0;
    [r, c] = find(depth_im>0.01); 
    b=sub2ind(size(depth_im),r, c);
    pix_z = depth_im(b);
    R = color_im(:,:,1);
    G = color_im(:,:,2);
    B = color_im(:,:,3);
    color = [R(b), G(b), B(b)];
    pix_x = (c - cam_intr(1,3)) .* pix_z / cam_intr(1,1);
    pix_y = (r - cam_intr(2,3)) .* pix_z / cam_intr(2,2);
    pixyz = [pix_x, pix_y, pix_z];
end