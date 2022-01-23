from utils.config import Config


data = {
    'a': 0,
    'b': 1,
    'c': 2,
}
config = Config(data)

assert config.a == data['a']
assert config.b == data['b']
assert config.c == data['c']
raised = False
try:
    config.z
except KeyError:
    raised = True
assert raised is True
