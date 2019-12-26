import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='balancebot-v0',
    entry_point='balance_bot.envs:BalancebotEnv',
)

register(
    id='balancebot-continuum-v0',
    entry_point='balance_bot.envs:BalancebotEnvContinuum',
)

register(
    id='balancebot-cpg-v0',
    entry_point='balance_bot.envs:BalancebotEnvCPG',
)

register(
    id='balancebot-noise-v0',
    entry_point='balance_bot.envs:BalancebotEnvNoise',
)
