import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='snakebot-v0',
    entry_point='snake_bot.envs:SnakebotEnv'
)