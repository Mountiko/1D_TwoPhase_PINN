using Plots
using ADCME
using PyCall
using DelimitedFiles
include("../simulation.jl")
include("../neural_net.jl")
np = pyimport("numpy")




const SRC_CONST = 86400.0 # turns seconds into days
const GRAV_CONST = 0.0    # gravity constant



################# --- Run Simulation --- #################


Nz = 1              # 1 cell in z direction
Nx = 101            # 100 cell in x direction   {  assert: typeof( (Nx-1) / (Nx_coloc-1) ) == Int  }
Nt = 2300           # number of timesteps       {  assert: typeof( (Nt) / (Nt_coloc-1) ) == Int  }
Δx = 30.0           # = Δz; cell size in x and z direction [m]
Δt = 0.1            # timespet size in [d]

z = (1:Nz)*Δx|>collect        # array of z coordinates
x = (1:Nx)*Δx|>collect        # array of x coordinates
X, Z = np.meshgrid(x, z)      # mesh grid

ρw = 1000.0     # density if wetting fluid [m^3/d]
ρo = 800.0      # density if produciton fluid [m^3/d]
μw = 1.0        # viscosity if wetting fluid [mPa*s]
μo = 10.7       # viscosity if production fluid [mPa*s]

g = GRAV_CONST  # gravitational acceleration

flow = 0.005 * (1/Δx^2)/10.0 * SRC_CONST # injection rate [m^3/d]

qw = zeros(Nt, Nz, Nx)      # injection rate
qw[:, 1, 1] .= flow
qo = zeros(Nt, Nz, Nx)      # produciton rate
qo[:,1,end] .= -flow

sw0 = zeros(Nz, Nx)         # initial saturation [%]

ϕ = 0.25 .* ones(Nz, Nx)    # "real" porosity field [%]
K = 20 .* ones(Nz, Nx)      # "real" permeability field [milidarcy]


println("\nRunning simulation...")

# turn all simulation parameters into tf.Tensors
tf_param_real = tfCtxGen(Nz, Nx, Δx, Nt, Δt, Z, X, ρw, ρo, μw, μo, K, g, ϕ, qw, qo, sw0, true)
# create computational graph for simulation
tf_sw_real, tf_p_real = imseq(tf_param_real)

sess = Session(); init(sess)                # initialise simulation session
sw_real = run(sess, tf_sw_real)             # get simulation result for saturation

sess.close()                                # close simulation session
println("\nSimulation session closed!")










################# --- Neural Network to approximate for simulation data --- #################

Nx_coloc = 51  # number of colocation points on x-axis
Nt_coloc = 101  # number of colocation points on time-axis

interpol_x = Int((Nx-1)/(Nx_coloc-1))
interpol_t = Int(Nt/(Nt_coloc-1))
coloc_x = 1:interpol_x:Nx+1         # colocation point indices in x
coloc_t = 1:interpol_t:Nt+1         # colocation point indices in time

x_dimless = range(0, 1, length=Nx)|>collect    # dimensionless spatial coordinates
t_dimless = range(0, 1, length=Nt+1)|>collect    # dimensionless time

X_star = repeat(x_dimless[coloc_x], Nt_coloc)                                # 1D array(Nx*(Nt+1)) of spatial mesh coordinates
T_star = collect(Iterators.flatten(repeat(t_dimless[coloc_t]', Nx_coloc)))   # 1D array(Nx*(Nt+1)) of time mesh

sw_real_star = collect(reshape(sw_real[coloc_t, 1, coloc_x]', Nt_coloc * Nx_coloc)) # flatten test data


ϵ = 2.5*10e-3                                   # 1/Pe: dimensionless diffusion coefficient
config = [20, 20, 20, 20, 20, 20, 20, 20, 1]    # neural network configuration


# get tf.Tensor struct of input parameters
tf_param_NN = tfCtxGenNN(X_star, T_star, sw_real_star, μw, μo, ϵ, config, Nx_coloc, Nt_coloc)

# neural network function
sw_nn, sw_t_nn, sw_xx_nn, f_w_x_nn = neural_net(tf_param_NN)

# loss function
loss, λ_1, λ_2, λ_3 = get_loss(tf_param_NN,
                                sw_nn,
                                sw_t_nn,
                                sw_xx_nn,
                                f_w_x_nn,
                                true, true, false, # calibrating λ1 and λ2, not calibrating λ3
                                true) # including production data in train data







################# --- training --- #################


# Adam optimizer
println("\nTraining with Adam optimizer...")
 
optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer_Adam.minimize(loss)

sess = Session(); init(sess)

epochs = 5000
losses = []

for i in 1:epochs
    run(sess, train_op)
    if i % 10 == 0
        println("Epoch: ", i, "\tLoss: ", run(sess, loss))
    end
    push!(losses, run(sess, loss))
end


println("\nTraining with BFGS optimizer...")
BFGS!(sess, loss, 10000)

name = "./model.mat" 
save(sess, name)

println("\nTraining finished!")








################# --- Computing errors --- #################


# get final loss and λs
loss_final, λ1, λ2, λ3 = run(sess, [loss, λ_1, λ_2, λ_3])

# get results
X_star_tf = constant(repeat(x_dimless, Nt+1))
T_star_tf = constant(collect(Iterators.flatten(repeat(t_dimless', Nx))))
sw_real_star = collect(reshape(sw_real[:, 1, :]', (Nt+1) * Nx))

sw_pred_star = run(sess, squeeze(fc(hcat(X_star_tf, T_star_tf),
                                            tf_param_NN.config, 
                                            tf_param_NN.Theta)) + 1)
# compute error
error = sum(@. sqrt((sw_pred_star - sw_real_star)^2)) / sum(@. sqrt(sw_real_star^2))


println("\nFinal loss:\t", loss_final)
println("L2 error:\t", error)
println("Lambda 1:\t", λ1)
println("Lambda 2:\t", λ2)
println("Lambda 3:\t", λ3)


sess.close()
println("\nTraining session closed!")








################# --- Plot results --- #################


println("\nPlotting results...")

sw_pred_plot = reshape(sw_pred_star, Nx, Nt+1)
sw_real_plot = reshape(sw_real_star, Nx, Nt+1)

# plot predicted and "real" saturation
for i in 1:Int((Nt)/5):Nt+1
    close("all")
    day = (i-1) * Δt
    println(day)
    Plots.plot(x, sw_real_plot[:, i], label="simulation")
    Plots.plot!(x, sw_pred_plot[:, i], label="NN", line=:dash)
    Plots.ylims!((0.0, 1.0))
    name = "./results/"*"$(day)"*".png"
    Plots.savefig(name)
end

# plot porosity
close("all")
Plots.plot(x, ϕ[1, :], label="Porosity real")
Plots.ylims!((0.0, 1.0))
name = "./results/poro_field.png"
Plots.savefig(name)

# plot permeability
close("all")
Plots.plot(x, K[1, :], label="Permeability real")
Plots.ylims!((0.0, 100.0))
name = "./results/perm_field.png"
Plots.savefig(name)

# plot losses 
close("all")
Plots.plot(1:epochs, losses, label="loss", yaxis=:log)
name = "./results/loss.png"
Plots.savefig(name)







################# --- Saving results --- #################


open("./results/error.txt", "w") do file
    write(file, "Final loss:\t $loss_final")
    write(file, "\nL2 error:\t $error")
    write(file, "\nLambda 1:\t $λ1")
    write(file, "\nLambda 2:\t $λ2")
    write(file, "\nLambda 3:\t $λ3")
    write(file, "\nNumber of colocation points:\t $(Nx_coloc*Nt_coloc)")
    write(file, "\nAdam epochs:\t $(epochs)")
end

open("./results/sw_real.txt", "w") do file
    writedlm(file, sw_real_plot')
end

open("./results/sw_pred.txt", "w") do file
    writedlm(file, sw_pred_plot')
end
