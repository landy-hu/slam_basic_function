function write2txt(filename, pt)
fid = fopen(filename,'w+');
[n,m]=size(pt);
fprintf(fid,'%d',n);
fprintf(fid,'%d',n);
fprintf(fid,'%d\n',n);
for i=1:n
    for j=1:m
        fprintf(fid,'%.4f  ', pt(i,j));
    end
            fprintf(fid,'\n');
end

fclose(fid);
end