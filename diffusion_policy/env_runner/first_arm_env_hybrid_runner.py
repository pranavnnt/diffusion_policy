import torch
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from dressing_policies.first_arm_env import FirstArmEnv  # ✅ your existing env

class FirstArmEnvRunner:
    """
    Hydra-compatible runner that evaluates a trained policy
    in your Unity FirstArmEnv (from dressing_policies repo).
    """
    def __init__(self,
                 unity_exe_path: str,
                 episodes: int = 5,
                 headless: bool = False,
                 device: str = "cuda:0",
                 max_steps: int = 500,
                 **kwargs):
        self.unity_exe_path = unity_exe_path
        self.episodes = episodes
        self.headless = headless
        self.device = torch.device(device)
        self.max_steps = max_steps

    def run(self, policy):
        env = FirstArmEnv(
            unity_exe_path=self.unity_exe_path,
            headless=self.headless,
            log_dir="./logs/eval_first_arm",
            is_demo=True,
        )

        returns = []
        with torch.no_grad():
            for ep in range(self.episodes):
                obs, _ = env.reset()
                done, ep_ret, steps = False, 0.0, 0

                # Buffers for hybrid obs
                state_buffer, image_buffer = [], []

                while not done and steps < self.max_steps:
                    # prepare obs tensors
                    state = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
                    image = torch.tensor(obs["image"].transpose(2,0,1)/255.0, dtype=torch.float32).unsqueeze(0)
                    depth = torch.tensor(obs["depth"], dtype=torch.float32).unsqueeze(0)

                    # concatenate image and depth if policy expects it
                    if "depth" in policy.normalizer.keys():
                        image = torch.cat([image, depth], dim=1)

                    obs_dict = {
                        "state": state.to(self.device),
                        "image": image.to(self.device),
                    }

                    out = policy.predict_action(obs_dict)
                    action = out["action"][0, 0].cpu().numpy()

                    obs, reward, done, _, _ = env.step(action)
                    ep_ret += reward
                    steps += 1

                print(f"[Episode {ep+1}] return={ep_ret:.2f}, steps={steps}")
                returns.append(ep_ret)

        env.close()
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "returns": returns,
        }
