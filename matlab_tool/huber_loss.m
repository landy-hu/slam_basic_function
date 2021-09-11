function errors = huber_loss(errors, delta)
    errors(errors<delta) = sqrt(0.5)*errors(errors<delta) ;
    errors(errors>delta) = sqrt(delta *errors(errors>delta) -0.5*delta^2);
end