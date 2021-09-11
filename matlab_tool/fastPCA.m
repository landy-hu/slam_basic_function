function [mat_mean,aff_mat]=fastPCA(Mat,ind)
% mat number of rows is the number of the samples, number of the column is
% the number of the feature dimmension.
% mat_mean is the mean of the feature.
% aff_mat is the affine matrix
% convariance matrix
 mat_mean=mean(Mat);
 Mat_off_mean=bsxfun(@minus,Mat,mat_mean);
if ind==1 %pca
   Cov_matrix=Mat_off_mean'*Mat_off_mean;
   [aff_mat,E]=eig(Cov_matrix);
   if det(aff_mat)==-1
       aff_mat = -aff_mat;
   end
   for i=1:size(aff_mat,2)
       aff_mat(:,i)=aff_mat(:,i)/norm(aff_mat(:,i));
   end
else %fastpca
    Cov_matrix=Mat_off_mean*Mat_off_mean';
    [aff_mat,E]=eig(Cov_matrix);
    aff_mat=Mat_off_mean'*aff_mat;
    for i=1:size(aff_mat,2)
       aff_mat(:,i)=aff_mat(:,i)/norm(aff_mat(:,i));
   end
end