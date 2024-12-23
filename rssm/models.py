import torch
import torch.nn as nn


class RSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._ans_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
        )
        self._ques_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
        )
        self._cell = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, prev_ans_emb, prev_hidden, ques_emb):
        encoded_ans = self._ans_encoder(prev_ans_emb)
        encoded_ques = self._ques_encoder(ques_emb)
        encoded_input = torch.concatenate([encoded_ans, encoded_ques], dim=-1)

        logit = self._cell(encoded_input, prev_hidden)
        return logit


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self._decoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, hidden_state):
        output = self._decoder(hidden_state)
        return output


class Scorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._ques_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ELU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ELU(),
        )
        self._scorer = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
            nn.ELU(),
        )
        self._predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 15), nn.Sigmoid()
        )

    def forward(self, ques_emb, tcs_emb, ans_emb, bans_emb=None):
        encoded_ques = self._ques_encoder(
            torch.concatenate([ques_emb, tcs_emb], dim=-1)
        )
        output_ans = self._scorer(torch.concatenate([encoded_ques, ans_emb], dim=-1))
        output_bans = self._scorer(torch.concatenate([encoded_ques, bans_emb], dim=-1))

        return self._predictor(torch.concatenate([output_ans, output_bans], dim=-1))


class LinearScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._predictor = nn.Sequential(
            nn.Linear(self.input_dim * 4, 15), nn.Hardsigmoid()
        )

    def forward(self, ques_emb, tcs_emb, ans_emb, bans_emb=None):
        return self._predictor(
            torch.concatenate([ques_emb, tcs_emb, ans_emb, bans_emb], dim=-1)
        )


class NaiveLinearScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._predictor = nn.Sequential(
            nn.Linear(self.input_dim * 2, 15), nn.Hardsigmoid()
        )

    def forward(self, ques_emb, tcs_emb):
        return self._predictor(torch.concatenate([ques_emb, tcs_emb], dim=-1))


class NaiveScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._ques_encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ELU(),
        )
        self._predictor = nn.Sequential(nn.Linear(self.hidden_dim, 15), nn.Sigmoid())

    def forward(self, ques_emb, tcs_emb):
        encoded_ques = self._ques_encoder(
            torch.concatenate([ques_emb, tcs_emb], dim=-1)
        )
        return self._predictor(encoded_ques)


class Vec2Latent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_features=16, class_size=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.class_size = class_size

        self._encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.num_features * self.class_size),
            nn.ELU(),
        )

    def get_stoch_state(self, logit):
        shape = logit.shape
        logit = torch.reshape(
            logit, shape=(*shape[:-1], self.num_features, self.class_size)
        )
        dist = torch.distributions.OneHotCategorical(logits=logit)
        stoch = dist.sample()
        stoch += dist.probs - dist.probs.detach()
        return torch.flatten(stoch, start_dim=-2, end_dim=-1)

    def forward(self, emb):
        logit = self._encoder(emb)
        return self.get_stoch_state(logit)


class RSSMV2(nn.Module):
    def __init__(self, ans_dim, ques_dim, hidden_dim=128):
        super().__init__()
        self.ans_dim = ans_dim
        self.ques_dim = ques_dim
        self.hidden_dim = hidden_dim
        self._ans_encoder = nn.Sequential(
            nn.Linear(self.ans_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
        )
        self._ques_encoder = nn.Sequential(
            nn.Linear(self.ques_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
        )
        self._cell = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, prev_ans_emb, prev_hidden, ques_emb):
        encoded_ans = self._ans_encoder(prev_ans_emb)
        encoded_ques = self._ques_encoder(ques_emb)
        encoded_input = torch.concatenate([encoded_ans, encoded_ques], dim=-1)

        logit = self._cell(encoded_input, prev_hidden)
        return logit
