import minari

names = [
  "D4RL/pointmaze/open-v2",
  "D4RL/pointmaze/umaze-v2",
  "D4RL/pointmaze/medium-v2",
  "D4RL/pointmaze/large-v2",
]

datasets = [minari.load_dataset(n, download=True) for n in names]  # download=True supported :contentReference[oaicite:3]{index=3}
print([d.total_episodes for d in datasets])
env = datasets[0].recover_environment()  # recover env from dataset :contentReference[oaicite:4]{index=4}
eps = datasets[0].sample_episodes(3)  # EpisodeData has observations/actions/rewards/terminations/truncations :contentReference[oaicite:5]{index=5}
e0 = eps[0]
print(e0.observations.keys(), e0.actions.shape)