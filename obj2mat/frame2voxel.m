% dir_root= '/home/lan/Downloads/obj2mat/tasks-all/';
% f_dir1 = dir(dir_root);
clear all
dir1 = '/home/mpl/chair/single-pass_release/output/03001627/';
dir1_save ='/home/mpl/chair/single-pass_release/output/03001627/03001627_one/'
% dir2 = '/home/mpl/chair/single-pass_release/shapenetcore.v2/03001627/';
% f_dir1 = dir(dir1);
% f_dir2 = dir(fullfile(dir2));
% dir1_save = '/home/mpl/Autoencoder/dataset/train_25d/voxel_grid_32/';
% dir2_save = '/home/mpl/Autoencoder/dataset/train/train_3d/voxel_grid_32/';
f_dir2 = dir(dir1);
fext = '*_depth.png';
count = 69444;
K = [30 0 240; 0 30 320; 0 0 1];
s = 500;
for t=1930:3700
%     for t=3701:3883
%     dir = [dir2,f_dir2(t).name,'/models/model_normalized.mat'];
%     voxel = load([dir2,f_dir2(t).name,'/models/model_normalized.mat']);
    diroot = [dir1, f_dir2(t).name];
    file_name = strcat(strcat('/03001627_',f_dir2(t).name),'_view')
%    dt = dir([diroot,fext]);
%     file_views = {dt.name};
%     dt = dir(fullfile(diroot));
        if ~exist([dir1_save,f_dir2(t).name])
    mkdir([dir1_save,f_dir2(t).name]);
else
    delete([dir1_save,f_dir2(t).name, '*'])
end
    for k =0:35
        train_25d = zeros(32,32,32);
        if k<10
            num = strcat('00',num2str(k));
        else
            num = strcat('0',num2str(k));
        end
        mat_name = [diroot,file_name, num,'.mat'];
        point = load(mat_name);

        save( [dir1_save,f_dir2(t).name, file_name,num,'.mat'], 'point')
        %% 
%         point = reshape(point.pos,640*480,3);
%         point(isnan(point(:,1)),:) =[];
% %         pcshow(pointCloud(point))
%         point=bsxfun(@minus, point, voxel.voxel.min);
%         point=point*voxel.voxel.max+1;
%         point = bsxfun(@plus, point, voxel.voxel.offset);
%         point = int8(round(point+0.5));
%         for i =1:size(point,1)
%             train_25d(point(i,1),point(i,2),point(i,3))=1;
%         end
%         count = count+1;
%         obj_write (train_25d, voxel.voxel.voxel,strcat(num2str(count),'.txt'),dir1_save,dir2_save)
        
%         point =[];
%         for k=1:32
%             [x,y] = find(train_25d(:,:,k)>0);
%             new = [x,y,(ones(size(x))*k)];
%             point = [point;new];
%         end
%         point(:,1) = -1 - 0.0625/2+0.0625*point(:,1);
%         point(:,2) = -1 - 0.0625/2+0.0625*point(:,2);
%         point(:,3) = -1 - 0.0625/2+0.0625*point(:,3);
%         figure(1)
%         
%         point(:,[2,3]) =point(:,[3,2]);
%         point(:,1) =-point(:,1);
%         pcshow(pointCloud(point))
%         xlabel('x')
%         ylabel('y')
%         zlabel('z')
%         axis([-1 1 -1 1 -1 1])
%         point =[];
%         for k=1:32
%             [x,y] = find(voxel.voxel.voxel(:,:,k)>0);
%             new = [x,y,(ones(size(x))*k)];
%             point = [point;new];
%         end
%         point(:,1) = -1 - 0.0625/2+0.0625*point(:,1);
%         point(:,2) = -1 - 0.0625/2+0.0625*point(:,2);
%         point(:,3) = -1 - 0.0625/2+0.0625*point(:,3);
%         figure(2)
%         pcshow(pointCloud(point))
%         xlabel('x')
%         ylabel('y')
%         zlabel('z')
%         axis([-1 1 -1 1 -1 1])
%         disp('hello!!!!')
    end
    
end
function vec = normalize(vector)
vec = vector./norm(vector);
end
function obj_write(train_25d, train_3d,name,dir_25d,dir_3d)
% dir_25d = '/home/lan/Documents/dataset/train/train_25d/';
% dir_3d = '/home/lan/Documents/dataset/train/train_3d/';
fid = fopen(strcat(dir_25d,name),'w');
for i=1:32
    for j=1:32
        for k=1:32
            if train_25d(i,j,k) ==1
                fprintf(fid, '%.0f',int8(i));
                fprintf(fid, '%.0f',int8(j));
                fprintf(fid, '%.0f\r\n',int8(k));
            else
                continue
            end
        end
    end
end
fclose(fid)
fid = fopen(strcat(dir_3d,name),'w');
for i=1:32
    for j=1:32
        for k=1:32
            if train_3d(i,j,k) ==1
                fprintf(fid, '%.0f',int8(i));
                fprintf(fid, '%.0f',int8(j));
                fprintf(fid, '%.0f\r\n',int8(k));
            else
                continue
            end
        end
    end
end
fclose(fid)
end
   