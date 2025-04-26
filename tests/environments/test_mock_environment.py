from neuro_fuzzy_multiagent.env.mock_environment import MockEnvironment


def test_mock_environment_step_and_reset():
    env = MockEnvironment()
    obs = env.reset()
    assert obs == 0
    for i in range(4):
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, int)
        assert reward == 1.0
        if i < 3:
            assert not done
        else:
            assert done
    obs = env.reset()
    assert obs == 0
