import numpy as np

# DROID & BRIDGE normalization constants
DROID_Q01 = np.array([
    -0.7776297926902771,
    -0.5803514122962952,
    -0.5795090794563293,
    -0.6464047729969025,
    -0.7041108310222626,
    -0.8895104378461838,
])
DROID_Q99 = np.array([
    0.7597932070493698,
    0.5726242214441299,
    0.7351000607013702,
    0.6705610305070877,
    0.6464948207139969,
    0.8897542208433151,
])

BRIDGE_Q01 = np.array([
    -0.02872725307941437,
    -0.04170349963009357,
    -0.026093858778476715,
    -0.08092105075716972,
    -0.09288699507713317,
    -0.20718276381492615,
    0.0
])
BRIDGE_Q99 = np.array([
    0.028309678435325586,
    0.040855254605412394,
    0.040161586627364146,
    0.08192047759890528,
    0.07792850524187081,
    0.20382574498653397,
    1.0
])

class TokenActionConverter:
    def __init__(self, q01, q99, n_action_bins=256, vocab_size=32000):
        self.q01 = np.array(q01)
        self.q99 = np.array(q99)
        self.vocab_size = vocab_size
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def action_to_token(self, actions):
        norm = 2 * (actions - self.q01) / (self.q99 - self.q01) - 1
        disc = np.array([np.abs(self.bin_centers - v).argmin() for v in norm])
        return self.vocab_size - disc - 1

    def token_to_action(self, tokens):
        disc = self.vocab_size - np.array(tokens)
        disc = np.clip(disc - 1, 0, len(self.bin_centers) - 1)
        norm = self.bin_centers[disc]
        return 0.5 * (norm + 1) * (self.q99 - self.q01) + self.q01


# Example DROID action (6D)
droid_action = np.array([0.3836, 0.0735, 0.5514, -2.8934, -0.1987, 0.1270])

# Initialize converters
droid_converter = TokenActionConverter(DROID_Q01, DROID_Q99)
bridge_converter = TokenActionConverter(BRIDGE_Q01[:6], BRIDGE_Q99[:6])

# Step 1: DROID action → tokens
droid_tokens = droid_converter.action_to_token(droid_action)

# Step 2: tokens → Bridge actions
bridge_actions = bridge_converter.token_to_action(droid_tokens)
bridge_actions = np.append(bridge_actions, 0.0)

bridge_actions
