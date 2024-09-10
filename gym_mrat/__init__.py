from gym.envs.registration import register

register(
    # id="gym_mrat/MRAT-v0",
    id="MRAT-v0",
    entry_point="gym_mrat.envs:MRAT_Env",
    max_episode_steps=10000,
    kwargs={

    }
)
