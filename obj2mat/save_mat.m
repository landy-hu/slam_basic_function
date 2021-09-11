function save_mat(savename, voxel)
	% just save the voxel to a file
	% used only to bypass the transparency requirement in parfor loop
	% disp(savename)
	save(savename, 'voxel');
end