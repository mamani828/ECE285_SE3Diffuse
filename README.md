# ECE285_SE3Diffuse

## Training

- For the Gaussian noise model, use `diffusion_model.py`.
- For correlated noise training, use `diffusion_model_correlated.py`.

When using correlated noise, specify the `rho` value with:
- `--noise_rho_train`
- `--noise_rho_sample`

## Evaluation

- `run_test_new_env.py` evaluates the test set with uncorrelated noise and returns the corresponding images.
- `run_correlated_ablations.py` runs all ablation experiments using correlated noise.
