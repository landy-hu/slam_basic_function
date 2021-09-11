function param = obj2voxel(filename, length)
    [v,f3,f4] = loadawobj(filename);
    obj.vertices = v';
    obj.faces = f3';
    if size(obj.vertices, 2) ~= 3 || size(obj.faces, 2) ~= 3
    	voxel = [];
    	warning(['Empty file:' filename])
    else
	    [voxel,param] = polygon2voxel(obj, [length length length],'auto',false,false);
	    % align 
	    voxel = permute(voxel, [1 3 2]);
        param.voxel=voxel;
	end
end