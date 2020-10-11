function [R_new, t_new, error] = robust_EM_ICP(pt1, pt2, initial_variabel, method)
    optimal_variabel = zeros(1,6);
    options = optimoptions(@lsqnonlin,  'Algorithm','levenberg-marquardt',  'MaxIterations', 20, 'StepTolerance',1e-06,'FunctionTolerance',1e-08);
    if strcmp(method{3,1},'icp')
            [optimal_variabel, error] = lsqnonlin(@icp_residual_term_simple, optimal_variabel, [], [], options, pt1, pt2, initial_variabel);
    elseif strcmp(method{3,1},'gma')
            corres = find_corr(pt1, pt2, initial_variabel, method);
           [optimal_variabel, error] = lsqnonlin(@residual_term_simple, optimal_variabel, [], [], options, pt1, pt2, initial_variabel, corres, method{5,1});
    end
     R_old = eul2rotm(initial_variabel(1:3));
     t_old = initial_variabel(4:6)';
     R = eul2rotm(optimal_variabel(1:3));
     t = optimal_variabel(4:6)';
     R_new = R*R_old;
     t_new = R*t_old+t;
end
function errors = icp_residual_term_simple(optimal_variabel, pt1, pt2, initial_variabel)
    R = eul2rotm(optimal_variabel(1:3));
    t = optimal_variabel(4:6)';
    R_old = eul2rotm(initial_variabel(1:3));
    t_old = initial_variabel(4:6)';
    pt2_old = (R_old*pt2'+t_old)';
    pt2 = (R*pt2_old'+t)';
    errors = icp_loss(pt1, pt2, pt2_old);
end

function errors = residual_term_simple(optimal_variabel, pt1, pt2, initial_variabel,corres, iteration)
    R = eul2rotm(optimal_variabel(1:3));
    t = optimal_variabel(4:6)';
    R_old = eul2rotm(initial_variabel(1:3));
    t_old = initial_variabel(4:6)';
    pt2_old = (R_old*pt2'+t_old)';
    pt2 = (R*pt2_old'+t)';
    errors = gma_error(pt1,pt2,pt2_old,corres, iteration);
%     [errors, ~] = do_trimming(errors, 0.9);
%     errors = huber_loss(errors);
end
function errors = icp_loss(pt1, pt2, pt2_old)
    [idx1, errors] = knnsearch(pt1, pt2, 'K',1);
    w = pt2_old -pt1(idx1,:);
    w = sum(w.*w,2);
    w = sqrt(exp(-5*w));
    errors = huber_loss(errors);
    errors = w.*errors;
end

function [error,I]=  do_trimming(error, num_ratio)
% B=error(I);
end_idx = uint32(num_ratio*size(error,1));
[~, I] = sort(error);
I=I(1:end_idx);
I = sort(I);
error = error(I);
end

function errors = huber_loss(errors)
   delta = 0.05;
    errors(errors<delta) = sqrt(0.5)*errors(errors<delta) ;
    errors(errors>delta) = sqrt(delta *errors(errors>delta) -0.5*delta^2);
end


