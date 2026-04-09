"""Today provider — single configurable source for 'what date is today.'

If SIM_TODAY env var is set, returns that date.
Otherwise, returns date.today().
No component should call datetime.now() directly.
"""

import os
from datetime import date


def get_today() -> date:
    """Return the effective 'today' date.

    In sim/test mode (SIM_TODAY set): returns the configured date.
    In production mode (SIM_TODAY unset): returns date.today().
    """
    sim_today = os.environ.get("SIM_TODAY")
    if sim_today:
        return date.fromisoformat(sim_today)
    return date.today()
