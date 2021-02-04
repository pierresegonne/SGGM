# From prior_gamma_per_pixel - for Fashion MNIST only
prior_α = 0.43274792054263117
prior_β = 0.005466103219719339
# ~1.567
prior_α = 1 + prior_β * (abs(prior_α - 1) / prior_β)
print(prior_α)