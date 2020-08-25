from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker
from .env_wrapper import EnvWrapper
from .lunar_lander_continous import LunarLanderContinous
from .beamforming import Beamforming

def create_env_wrapper(config):
    env_name = config['env']
    if env_name == "Beamforming":
    	return Beamforming(config)
    return EnvWrapper(env_name)