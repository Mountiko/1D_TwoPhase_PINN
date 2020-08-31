using Plots
using DelimitedFiles
using PyCall
include("../simulation.jl")
np = pyimport("numpy")



const SRC_CONST = 86400.0 # turns seconds into days
const GRAV_CONST = 0.0    # gravity constant
const CONV_CONST = 2000   # linear relation between permeability and porosity



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


ϕ_real = ones(Nz, Nx)                # "real" porosity field [%]
ϕ_real[1, :] = ([610.0, 615.0, 620.0, 623.0, 628.0, 631.0, 635.0, 637.0, 638.0,
                640.0, 108.0, 111.0, 113.0, 120.0, 123.0, 125.0, 130.0, 139.0,
                145.0, 151.0, 155.0, 162.0, 515.0, 511.0, 506.0, 503.0, 498.0,
                495.0, 490.0, 910.0, 901.0, 892.0, 883.0, 875.0, 867.0, 859.0,
                852.0, 845.0, 839.0, 710.0, 709.0, 707.0, 704.0, 700.0, 695.0,
                689.0, 682.0, 674.0, 665.0, 655.0, 644.0, 633.0, 622.0, 610.0,
                689.0, 682.0, 674.0, 665.0, 655.0, 644.0, 633.0, 622.0, 610.0,
                610.0, 610.0, 610.0, 610.0, 630.0, 640.0, 650.0, 660.0, 310.0,
                310.0, 310.0, 330.0, 320.0, 330.0, 330.0, 340.0, 345.0, 350.0,
                355.0, 360.0, 365.0, 369.0, 373.0, 377.0, 381.0, 384.0, 386.0,
                388.0, 689.0, 388.0, 386.0, 384.0, 381.0, 377.0, 373.0, 369.0,
                365.0, 360.0])./ CONV_CONST
K_real = ones(Nz, Nx)
K_real[1, :] = ϕ_real[1, :] .* CONV_CONST       # "real" permeability field [milidarcy]


ϕ_init = minimum(ϕ_real) .* ones(Nz, Nx)    # initial porosity guess




# to compute error in longer simulation than what NN predicted change Nt_dob
Nt_dob = Nt
qw_dob = zeros(Nt_dob, Nz, Nx)      # injection rate
qw_dob[:, 1, 1] .= flow
qo_dob = zeros(Nt_dob, Nz, Nx)      # produciton rate
qo_dob[:,1,end] .= -flow

tf_param_real = tfCtxGen(Nz, Nx, Δx, Nt_dob, Δt, Z, X, ρw, ρo, μw, μo, K_real, g, ϕ_real, qw_dob, qo_dob, sw0, true)
tf_sw_real, tf_p_real = imseq(tf_param_real)





println("\nInverting field parameters ...")

# turn all simulation parameters into tf.Tensors
tf_param_init = tfCtxGen(Nz, Nx, Δx, Nt, Δt, Z, X, ρw, ρo, μw, μo, 0, g, ϕ_init, qw, qo, sw0, false)
# create computational graph for simulation
tf_sw_init, tf_p_init = imseq(tf_param_init)




# reading neural network predicted saturation data
sw_pred = readdlm("./results/sw_pred.txt")
sw_pred_tf = constant(reshape(sw_pred, Nt+1, Nz, Nx))



loss = sum((tf_sw_init - sw_pred_tf)^2) / (Nt+1 * Nx)



# Adam optimzer
optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer_Adam.minimize(loss)

sess = Session(); init(sess)

sw_real = run(sess, tf_sw_real)


sw_error = 1.0
ϕ_error = 1.0
K_error = 1.0
ϕ_inv = ϕ_init
K_inv = ϕ_init * CONV_CONST
opt_iter = 0
sw_errs =[]

epochs = 100
losses = []

for i in 1:epochs
    run(sess, train_op)
    if i % 1 == 0
        print("Epoch: ", i, "\t Loss: ", run(sess, loss))

        ϕ_tmp = run(sess, tf_param_init.ϕ)
        K_tmp = run(sess, tf_param_init.K)

        tf_param_inv = tfCtxGen(Nz, Nx, Δx, Nt_dob, Δt, Z, X, ρw, ρo, μw, μo, K_tmp, g, ϕ_tmp, qw_dob, qo_dob, sw0, true)
        tf_sw_inv, tf_p_inv = imseq(tf_param_inv)
        
        sess_tmp = Session(); init(sess_tmp)
        sw_inv = run(sess_tmp, tf_sw_inv)
        sess_tmp.close()

        
        sw_error_tmp = sum(@. sqrt((sw_real - sw_inv)^2)) / sum(@. sqrt(sw_real^2))
        println("\tsw error: $sw_error_tmp")

        global sw_error
        global ϕ_error
        global K_error
        global ϕ_inv
        global K_inv
        if sw_error_tmp < sw_error
            ϕ_error_tmp = sum(@. sqrt((ϕ_real - ϕ_tmp)^2)) / sum(@. sqrt(ϕ_real^2))
            K_error_tmp = sum(@. sqrt((K_real - K_tmp)^2)) / sum(@. sqrt(K_real^2))

            sw_error = sw_error_tmp
            ϕ_error = ϕ_error_tmp
            K_error = K_error_tmp

            ϕ_inv = ϕ_tmp
            K_inv = K_tmp
            opt_iter = i
        end

        push!(sw_errs, sw_error_tmp)
    end
    push!(losses, run(sess, loss))
end
# Adam optimzer


sess.close()                                # close simulation session
println("\nInversion finished!")

println("\n\nBest results at epoch $opt_iter")
println("sw error:\t $sw_error")
println("ϕ error:\t $ϕ_error")
println("K error:\t $K_error")


# write results to file

open("./results/inv_results/inv_error.txt", "w") do file
    write(file, "sw error:\t $sw_error")
    write(file, "\nϕ error:\t $ϕ_error")
    write(file, "\nK error:\t $K_error")
    write(file, "\nAdam epochs:\t $(epochs)")
    write(file, "\nBest results at epoch $opt_iter")
end

open("./results/inv_results/inv_poro.txt", "w") do file
    writedlm(file, ϕ_inv)
end

open("./results/inv_results/inv_perm.txt", "w") do file
    writedlm(file, K_inv)
end



# get simulation with best field values
tf_param_inv = tfCtxGen(Nz, Nx, Δx, Nt_dob, Δt, Z, X, ρw, ρo, μw, μo, K_inv, g, ϕ_inv, qw_dob, qo_dob, sw0, true)
tf_sw_inv, tf_p_inv = imseq(tf_param_inv)

sess = Session(); init(sess)
sw_inv = run(sess, tf_sw_inv)
sess.close()

# plot predicted and "real" saturation
for i in 1:Int((Nt_dob)/10):Nt_dob+1
    close("all")
    day = (i-1) * Δt
    println(day)
    Plots.plot(x, sw_real[i, 1, :], label="simulation")
    if i < Nt
        Plots.plot!(x, sw_pred[i, :], label="NN", line=:dash, color=:orange)
    end
    Plots.plot!(x, sw_inv[i, 1, :], label="inverted", line=:dash, color=:green)
    Plots.ylims!((0.0, 1.0))
    name = "./results/inv_results/inv_$(day).png"
    Plots.savefig(name)
end


# plot inverted porosity
close("all")
Plots.plot(x, ϕ_real[1, :], label="Porosity real")
Plots.plot!(x, ϕ_inv[1, :], label="Porosity inverted")
Plots.ylims!((0.0, 1.0))
name = "./results/inv_results/inv_poro_field.png"
Plots.savefig(name)

# plot inverted permeability
close("all")
Plots.plot(x, K_real[1, :], label="Permeability real")
Plots.plot!(x, K_inv[1, :], label="Permeability inverted")
Plots.ylims!((0.0, 1000.0))
name = "./results/inv_results/inv_perm_field.png"
Plots.savefig(name)

# plot inversion losses
close("all")
Plots.plot(1:epochs, losses, label="inversion loss", yaxis=:log)
name = "./results/inv_results/inv_loss.png"
Plots.savefig(name)



# plot inversion error
close("all")
Plots.plot(1:epochs, sw_errs, label="inversion errors", yaxis=:log)
name = "./results/inv_results/inv_errors.png"
Plots.savefig(name)