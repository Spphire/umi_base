import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vq_bet.vq_behavior_transformer.bet import BehaviorTransformer
from diffusion_policy.model.vq_bet.vq_behavior_transformer.utils import MLP

class VqBetImagePolicy(BehaviorTransformer, BaseImagePolicy):
    def __init__(self, **kwargs):
        kwargs['act_dim'] = kwargs['shape_meta']['action']['shape'][0]
        low_dim_keys = list()
        key_shape_map = dict()
        for key, attr in kwargs['shape_meta']['obs'].items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'low_dim':
                low_dim_keys.append(key)
        del kwargs['shape_meta']

        super(VqBetImagePolicy, self).__init__(**kwargs)
        self._resnet_header = MLP(
            in_channels=512 + sum([key_shape_map[key][0] for key in low_dim_keys]),
            hidden_channels=[1024],
        )
        self._vqvae_model.eval()

        self.n_obs_steps = kwargs['obs_window_size']
        self.horizon = kwargs['act_window_size']
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map

        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss_and_metric(self, batch):
        obs_seq = batch['obs']['left_wrist_img']
        extra_obs_seq = torch.cat([self.normalizer[key].normalize(batch['obs'][key]) for key in self.low_dim_keys],dim=-1)
        goal_seq = None
        action_seq = self.normalizer['action'].normalize(batch['action'])

        predicted_action, loss, loss_dict = self.forward(obs_seq, extra_obs_seq, goal_seq, action_seq)

        loss_dict['loss'] = loss
        return loss_dict

    def predict_action(self, obs_dict):
        obs_seq = obs_dict['left_wrist_img']
        extra_obs_seq = torch.cat([self.normalizer[key].normalize(obs_dict[key]) for key in self.low_dim_keys], dim=-1)
        goal_seq = None
        action_seq = None

        predicted_action, _, _ = self.forward(obs_seq, extra_obs_seq, goal_seq, action_seq)

        predicted_action = self.normalizer['action'].unnormalize(predicted_action)

        result = {
            'action': predicted_action[[-1]],
            'action_pred': predicted_action,
        }
        return result

    def reset(self):
        pass