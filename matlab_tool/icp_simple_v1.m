function [R_new, t_new, error] = icp_simple_v1(pt1, pt2, older_V)
    optimal_V = zeros(1,6);
    options = optimoptions(@lsqnonlin,  'Algorithm','levenberg-marquardt',  'MaxIterations', 20, 'StepTolerance',1e-08,'FunctionTolerance',1e-10);

    [optimal_V, error] = lsqnonlin(@icp_residual_term_simple, optimal_V, [], [], options, pt1, pt2, older_V);

     R_old = eul2rotm(older_V(1:3));
     t_old = older_V(4:6)';
     R = eul2rotm(optimal_V(1:3));
     t = optimal_V(4:6)';
     R_new = R*R_old;
     t_new = R*t_old+t;
end
function errors = icp_residual_term_simple(optimal_V, pt1, pt2, older_V)
    R = eul2rotm(optimal_V(1:3));
    t = optimal_V(4:6)';
    R_old = eul2rotm(older_V(1:3));
    t_old = older_V(4:6)';
    pt2_old = (R_old*pt2'+t_old)';
    pt2 = (R*pt2_old'+t)';
    [idx1, errors] = knnsearch(pt1, pt2, 'K',1);
    errors = huber_loss(errors);
end
function errors = huber_loss(errors)
    delta = 0.02;
    temp = errors;
    errors(temp<=delta) = sqrt(0.5)*errors(temp<=delta) ;
    errors(temp>delta) =sqrt(delta *errors(temp>delta) -0.5*delta^2);
end
