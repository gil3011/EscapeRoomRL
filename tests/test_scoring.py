from engine.scoring import discounted_return


def test_discounted_return_undiscounted_is_a_plain_sum():
    assert discounted_return([1.0, 2.0, 3.0], gamma=1.0) == 6.0


def test_discounted_return_weights_later_rewards_less():
    # G = r0 + gamma*r1 + gamma^2*r2
    g = discounted_return([10.0, 10.0, 10.0], gamma=0.5)
    assert abs(g - (10.0 + 0.5 * 10.0 + 0.25 * 10.0)) < 1e-9


def test_discounted_return_reaching_goal_sooner_scores_higher():
    # same eventual reward, but reached one step later -- should discount to less.
    sooner = discounted_return([0.0, 100.0], gamma=0.9)
    later = discounted_return([0.0, 0.0, 100.0], gamma=0.9)
    assert sooner > later


def test_discounted_return_empty_episode_is_zero():
    assert discounted_return([], gamma=0.9) == 0.0
