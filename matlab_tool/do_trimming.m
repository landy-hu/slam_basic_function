function [error,I]=  do_trimming(error, num_ratio)
% B=error(I);
end_idx = uint32(num_ratio*size(error,1));
[~, I] = sort(error);
I=I(1:end_idx);
I = sort(I);
error = error(I);
end
