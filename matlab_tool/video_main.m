clear all
data_path = "/media/lan/Samsung_T5/scannet/scans/";
seq = "scene0015_00";
rgb_name=data_path + seq+  "/associations/global_rgb.txt";
rgb_list = {};
fid = fopen(rgb_name);
tline = fgetl(fid);
count=1;
while ischar(tline)
    rgb_list{count} = tline;
    tline = fgetl(fid);
    count = count +1;  
end
fclose(fid);

pt_path = data_path + seq+  "/map/";
mask_path = data_path + seq+  "/seg_out/seg/";
outputVideo = VideoWriter( 'scene0015_00.avi' );
outputVideo.FrameRate = 12;
open(outputVideo)
figure(1)
set(gcf,'position',[100,100,1800,900]);
set(gcf,'color',[1 1 1])
view_vector =[-0.3 -0.1 0.3];

for i =1:count-1
    
     rgb_name = rgb_list{1,i};
     rgb_name = strsplit(rgb_name,  '/');
     rgb_name = rgb_name{1, 9};
     pt_name = pt_path + num2str(i) + '.mat';
     mask_name = mask_path + rgb_name(1:end-3) + 'jpg';
     if ~exist(pt_name, 'file')
         continue
     end
    if ~exist(mask_name, 'file')
        continue
    end
     mask = imread(mask_name);
     subplot(1,2,1)
     imshow(mask)
     set(gca,'position',[0.01 0.05 0.4 0.9])
     load(pt_name);
     ptc = pointCloud(pt(:,1:3), 'Color', pt(:,4:6));
     subplot(1,2,2)
     pcshow(ptc,'MarkerSize' , 30)
     view(view_vector)
     set(gca,'position',[0.42 0.05 0.57 0.95])
     f = getframe(gcf);
     writeVideo(outputVideo,f.cdata)
end
close(outputVideo);