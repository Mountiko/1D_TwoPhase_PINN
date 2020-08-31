using ADCME
using Plots
using PyCall
@pyimport scipy.interpolate as si
pyplot()




################# --- functions --- #################


# flux function non-convex
function f_w(sw, M=1)
    # u := saturation of water
    # M := ratio of viscosities
    return sw^2 / (sw^2 + ((1 - sw)^2 / M))
end



# gradient of non-convex flux
function f_w_grad(sw, M=1)
    return 2 * M * sw * (1-sw) / ((M+1) * sw^2 - 2 * sw + 1)^2
end



# analytical Buckley-Leverett solution to sw (sw := Saturation of water)
function sw(x, t, M=1)
    # x := spacial coord
    # t := time nodes
    # M := viscosity ratio μ_oil/μ_water
    s = range(0, 1, length=length(x))[end:-1:1]
    x_s = @. f_w_grad(s)

    # compute sasturation before shock
    dif = x_s[2] - x_s[1]
    for i in 2:length(x)
        if x_s[i+1] - x_s[i] < dif
            x_s = x_s[1:i]
            @goto stop
        else
            dif = x_s[i+1] - x_s[i]
        end
    end
    @label stop

    x_s_times = []
    [push!(x_s_times, x_s * k) for k in t ]

    sw_ = zeros(Nx*Nt)

    # Perform the interpolation
    for k in 1:length(t)
        bench = x_s_times[k][end]
        tmp = si.griddata(Array(x_s_times[k]), Array(s), Array(x))
        sw_[(k-1)*Nx+1:k*Nx] = @. ifelse(x <= bench, tmp, 0.0)
        convert(Array{Float64,1}, x_s_times[k])
    end

    return sw_
end



### --- numerical parameters --- ###


X = 1 # spacial domain from 0 to 1
T = 1 # time domain from 0 to 1

dx = 0.01
dt = 0.01
Nx = round(Int, X/dx + 1) # spacial nodes
Nt = round(Int, T/dt) # time nodes

# simple coordinates
x = range(0, X, step=dx) # x coords
t = range(dt, T, step=dt) # t steps

# 1D array of all possible coordinte pairs
# will also be all colocation points for PDE regularization
x_star = repeat(x, Nt)
t_star = collect(Iterators.flatten(repeat(t', Nx)))




################# --- anal solution --- #################


sw_anal = sw(x, t)
sw_anal_tf = constant(sw_anal)




################# --- neural net --- #################


# training input for neural network
x_star_tf = constant(x_star)
t_star_tf = constant(t_star)
#train_input = constant(hcat(train_XX, train_TT))

# layers
config = [20, 20, 20, 20, 20, 20, 20, 20, 1]
# xavier initialization
θ = Variable(fc_init([2; config]))

# neural network function
sw_nn = squeeze(fc(hcat(x_star_tf, t_star_tf), config, θ)) + 1

# derivatives for PDE
sw_t_nn = tf.gradients(sw_nn, t_star_tf)[1]
sw_x_nn = tf.gradients(sw_nn, x_star_tf)[1]
sw_xx_nn = tf.gradients(sw_x_nn, x_star_tf)[1]
f_w_nn = f_w(sw_nn)
f_w_x_nn = tf.gradients(f_w_nn, x_star_tf)[1]



### --- loss --- ###


# loss funciton: (IC/BC - analitical_solution) + (sw_t + f_w(sw) - ϵ * sw_xx)
# sw_t       := time derivative of water saturation 
# f_w(sw)    := flux funciton dependend on water saturation
# ϵ         := 1/Pe
# sw_xx      := second spatial derivative of water saturation

ϵ = constant(2.5*10e-3)

# Loss with diffusion
loss = (sum((sw_nn[1:Nx] - sw_anal_tf[1:Nx])^2) / Nx + 
        sum((sw_nn[Nx+1:Nx:end] - sw_anal_tf[Nx+1:Nx:end])^2)/ (Nt - 1) + 
        sum((sw_t_nn + f_w_x_nn - ϵ * sw_xx_nn)^2) / (Nx * Nt))
# Loss without diffusion
#= loss = (sum((u_nn[1:Nx] - u_anal_train[1:Nx])^2) / Nx + 
        sum((u_nn[Nx+1:Nx:end] - u_anal_train[Nx+1:Nx:end])^2)/ (Nt - 1) + 
        sum((u_nn[Nx*2-1:Nx:end] - u_anal_train[Nx*2-1:Nx:end])^2)/ (Nt - 1) + 
        sum((u_t_nn + f_w_x_nn)^2) / (Nx * Nt)) =#





################# --- training --- #################


### optimising


losses = []
# callback function for optimisation
i = 0
function cb()
    global i
    if i % 10 == 0
        println("Epoch: $i")
    end
    #push!(losses, loss)
    i += 1
end

sess = Session(); init(sess)


# Scipy optimizer
opt = ScipyOptimizerInterface(loss; method="L-BFGS-B", options=Dict("maxiter"=> 3000, "ftol"=>1e-12, "gtol"=>1e-12))
ScipyOptimizerMinimize(sess, opt; loss_callback=cb)

#BFGS!(sess, loss)



### --- predict --- ###

sw_test_nn = run(sess, squeeze(fc(hcat(x_star_tf, t_star_tf), config, θ))+1)

error = sum(@. sqrt((sw_test_nn - sw_anal).^2)) / sum(@. sqrt(sw_anal.^2))
println("\nFinal Loss:\t", run(sess, loss))
println("sw error:\t", error)





### --- Plots --- ###


println("\nPrinting images...")
close("all")


# times at which to plot
t_plot = [0.25, 0.5, 0.75]
sw_anal = reshape(sw_anal, (Nx, Nt))
sw_test_nn = reshape(sw_test_nn, (Nx, Nt))

for i in 1:length(t_plot)
    idx = round(Int, t_plot[i] / dt * Nt*dt)
    println("0.$idx")
    plot(x, sw_anal[:, idx], label="anal")
    plot!(x, sw_test_nn[:, idx], label="NN", line=:dash)
    name = "./results/0.$idx.png"
    savefig(name)
end



### --- Save NN --- ###


println("saving NN...")
# note if you run script multiple times -> number epochs adds up
name = "./model.mat" 
save(sess, name)


sess.close()
