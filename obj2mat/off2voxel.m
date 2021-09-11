function param = off2voxel(filename, length)
    obj = off_loader(filename,0);
    [voxel,param] = polygon2voxel(obj, [length length length], 'auto',true,false);
    
    % align 
    voxel = flip(voxel, 1);
    param.voxel = voxel;
end