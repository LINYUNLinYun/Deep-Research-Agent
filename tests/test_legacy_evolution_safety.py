from src.evolution.judge import Judge


def test_reward_shaping_maps_zero_midpoint_and_maximum_without_saturation():
    judge = Judge(policy=None)

    assert judge.shape_reward({}) == -1.0
    assert judge.shape_reward({"factual_accuracy": 5.0}) < 1.0
    assert judge.shape_reward({
        "factual_accuracy": 10.0,
        "coverage": 10.0,
        "logical_coherence": 10.0,
        "citation_quality": 10.0,
        "efficiency": 10.0,
    }) == 1.0
