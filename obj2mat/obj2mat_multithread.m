task_dir = 'tasks-all_view';
task_prefix = 'task';
task_list = dir(task_dir);
tasks_num = size(task_list, 1);

p = gcp('nocreate');
if isempty(p)
  parpool;
end

len = 64;

parfor ind = 1:tasks_num
	task = task_list(ind).name;
	if size(task,2) >= size(task_prefix,2) && strcmp(task(1:size(task_prefix,2)), task_prefix)
		% this is a task
		% get a list of files
		taskfull = fullfile(task_dir, task);
		fid = fopen(taskfull);
		filename = fgetl(fid);
		while ischar(filename)
			ext = filename(end-2:end);
			if strcmp(ext, 'off')
				% disp([num2str(ind) ': ' filename])
				voxel = off2voxel(filename, len);
			elseif strcmp(ext, 'obj')
				% disp([num2str(ind) ': ' filename])
				voxel = obj2voxel(filename, len);
			else
				disp('Unsupported file extension');
            end
            vol = voxel.voxel;
            xyz = [];
            for k =1:size(vol,1)
                [x,y] = find(vol(:,:,k)==1);
                xyz = [xyz; [x,y,k*ones(size(x,1),1)]];
            end
            xyz = xyz(:,[1,3,2]);
           
            xyz = xyz-voxel.offset;
            xyz = (xyz-1)/voxel.scaling;
            xyz = xyz+voxel.min;
%             xyz = xyz*eul2rotm([-pi/2,0,0], 'XYZ')' ;
%             xyz = xyz - mean(xyz);
            
            %pcshowpair(pointCloud(xyz), pointCloud(voxel.pt))
%             xlabel('x')
%             ylabel('y')
%             zlabel('z')
         

            
			if size(voxel.voxel) == len
% 				savename = [filename(1:end-4) , num2str(len) '.mat'];
				% disp(savename)
% 				save_mat(savename, xyz);
				savename = [filename(1:end-4) , num2str(len) '.txt'];

                write2txt(savename, xyz );
			else
				warning('Empty file found.')
			end

			filename = fgetl(fid);
		end
	end
	disp(['task complete: ' num2str(ind) '/' num2str(tasks_num)])
end




