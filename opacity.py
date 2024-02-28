from scene.gaussian_model import GaussianModel
import matplotlib.pyplot as plt
from torch import nn

gaussian = GaussianModel(sh_degree=3)
path = './output/family/point_cloud/iteration_30000/point_cloud.ply'
gaussian.load_ply(path)

opacity = gaussian.get_opacity
opacity = opacity.detach().cpu().numpy()

plt.hist(opacity, bins=50, density=False, histtype='bar', color = 'black')
plt.title('family opacity histogram')
plt.show()

# min_opacity = 0.1
# prune_mask = (opacity > min_opacity).squeeze()

# gaussian._xyz = nn.Parameter(gaussian._xyz[prune_mask].requires_grad_(True))
# gaussian._features_dc = nn.Parameter(gaussian._features_dc[prune_mask].requires_grad_(True))
# gaussian._features_rest = nn.Parameter(gaussian._features_rest[prune_mask].requires_grad_(True))
# gaussian._opacity = nn.Parameter(gaussian._opacity[prune_mask].requires_grad_(True))
# gaussian._scaling = nn.Parameter(gaussian._scaling[prune_mask].requires_grad_(True))
# gaussian._rotation = nn.Parameter(gaussian._rotation[prune_mask].requires_grad_(True))

# gaussian.save_ply('./output/point_cloud.ply')
