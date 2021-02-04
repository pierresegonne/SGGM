# From prior_gamma_per_pixel - for MNIST only
prior_α = 0.3927581767454884
prior_β = 0.0010704155012337458
# ~1.625
prior_α = 1 + prior_β * (abs(prior_α - 1) / prior_β)
