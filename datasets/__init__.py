from datasets.strategy import *

STRATEGY_MAPPING = {
    "interval_average": interval_average_strategy,
    "interval_ensemble": interval_ensemble_strategy,
    "episodic_ensemble": interval_ensemble_strategy,
    "episodic_average": episodic_average_strategy,
    "minmax": minmax_strategy,
    "transformer_decompose": transformer_decompose_strategy,
    "pg_reshaping": pg_reshaping_strategy,
    "scale": scale_strategy,
}
