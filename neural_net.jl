using ADCME

function neural_net(args)
    #= 
    x:      flattened mesh coordinates x-direction (PyObject: tf.Tensor)
    t:      flattened mesh coordinates time (PyObject: tf.Tensor)
    μ_w:    viscosity of wetting fluid (PyObject: tf.Tensor)
    μ_o:    viscosity of production fluid (PyObject: tf.Tensor)
    config: configuration of neural network i.e. [20, 20, 20, 3] (PyObject: tf.Tensor)
    θ:      already initialised weights and biases
    =#
    
    sw_nn = squeeze(fc(hcat(args.x, args.t), args.config, args.Theta)) + 1  # return neural network result

    sw_t_nn = tf.gradients(sw_nn, args.t)[1]             # first time derivative of saturation
    sw_x_nn = tf.gradients(sw_nn, args.x)[1]             # first spatial derivative of saturation
    sw_xx_nn = tf.gradients(sw_x_nn, args.x)[1]          # second spatial derivative of saturation

    λ_w = sw_nn^2 / args.μ_w                             # mobility wetting fluid
    λ_o = (1 - sw_nn)^2 / args.μ_o                       # mobility production fluid

    f_w_nn = λ_w / (λ_w + λ_o)                      # flux-function
    f_w_x_nn = tf.gradients(f_w_nn, args.x)[1]           # spatial derivative of flux-function

    return sw_nn, sw_t_nn, sw_xx_nn, f_w_x_nn
end




function get_loss(args, sw_pred, sw_t_pred, sw_xx_pred, f_w_x_pred, calib1, calib2, calib3, prod)
    #= 
    sw_train:       training saturation (PyObject: tf.Tensor)
    sw_pred:        predicted saturation (PyObject: tf.Tensor)
    sw_t_pred:      predicted saturation derivative with respect to time (PyObject: tf.Tensor)
    sw_xx_pred:     predicted saturation second derivative with respect to space (PyObject: tf.Tensor)
    f_w_x_pred:     predicted flux-fluncion result derivative with respect to space (PyObject: tf.Tensor)
    ϵ:              1/Pe (PyObject: tf.Tensor)
    calib1:         calibrate saturation term (Bool)
    calib2:         calibrate flux-funcion term (Bool)
    calib3:         calibrate diffusion term (Bool)
    prod:           include production data in loss function (Bool)
    =#

    if calib1                   # define first calibration parameter
        λ_1 = Variable(1.0)
    else
        λ_1 = constant(1.0)
    end

    if calib2                   # define second calibration parameter
        λ_2 = Variable(1.0)
    else
        λ_2 = constant(1.0)
    end

    if calib3                   # define third calibration parameter
        λ_3 = Variable(1.0)
    else
        λ_3 = constant(1.0)
    end


    # compute loss

    # initial loss
    loss_init = sum((sw_pred[1 : args.Nx] - args.sw[1 : args.Nx])^2) / args.Nx
    # injection loss
    loss_inj =  sum((sw_pred[1 : args.Nx : (args.Nx * args.Nt)] -
                                args.sw[1 : args.Nx : (args.Nx * args.Nt)])^2) / args.Nt
    # production loss
    loss_prod = sum((sw_pred[args.Nx : args.Nx : (args.Nx * (args.Nt))] -
                                args.sw[args.Nx : args.Nx : (args.Nx * (args.Nt))])^2)/ args.Nt
    # PDE regularization
    loss_PDE = sum((sw_t_pred * λ_1 + f_w_x_pred * λ_2 - ϵ * sw_xx_pred * λ_3)^2) / (args.Nx * args.Nt)

    # full loss
    if prod
        loss = loss_init + loss_inj + loss_prod + loss_PDE
    else
        loss = loss_init + loss_inj + loss_PDE
    end

    return loss, λ_1, λ_2, λ_3
    
end




mutable struct CtxNN
    x; t; sw; μ_w; μ_o; ϵ; Theta; config; Nx; Nt
  end

function tfCtxGenNN(x, t, sw, μ_w, μ_o, ϵ, config, Nx, Nt)
    x_tf = constant(x)                          # tf.Tensor of spatial mesh
    t_tf = constant(t)                          # tf.Tensor of time mesh
    μ_w_tf = constant(μ_w)                      # tf.Tensor of wetting viscosity
    μ_o_tf = constant(μ_o)                      # tf.Tensor of production viscosity
    Θ_tf = Variable(fc_init([2; config]))       # xavier initialise weights and biases
    sw_tf = constant(sw)                        # tf.Tensor of test data
    ϵ_tf = constant(ϵ)                          # tf.Tensor of dimensionless diffusion coefficient

    return CtxNN(x_tf, t_tf, sw_tf, μ_w_tf, μ_o_tf, ϵ_tf, Θ_tf, config, Nx, Nt)
end
