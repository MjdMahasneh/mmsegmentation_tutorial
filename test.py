import torch
import mmcv
import matplotlib.pyplot as plt
from mmengine import Config
from mmseg.registry import TRANSFORMS
from mmseg.apis import init_model, inference_model, show_result_pyplot

# **Paths**
img_dir = './iccv09Data/images'
img_path = f'{img_dir}/6000127.jpg'

# **Use the config from training**
# Working directory & config path
work_dir = './work_dirs/tutorial'
cfg_path = f'{work_dir}/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'

# Model checkpoint path
checkpoint_path = f'{work_dir}/iter_5000.pth'

# **Load training config**
cfg = Config.fromfile(cfg_path)
cfg.model.test_cfg = dict(mode='whole')

# # ** Build the test pipeline without LoadAnnotations**
# test_pipeline = []
# for transform_cfg in cfg.test_pipeline:
#     if transform_cfg['type'] != 'LoadAnnotations':  # **REMOVE LoadAnnotations**
#         test_pipeline.append(TRANSFORMS.build(transform_cfg))


if __name__ == "__main__":
    # **Initialize model using correct trained config**
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = init_model(cfg, checkpoint_path, 'cuda:0')

    # Load and process image
    img = mmcv.imread(img_path)

    # Run inference
    result = inference_model(model, img)

    # Display result
    plt.figure(figsize=(8, 6))
    vis_result = show_result_pyplot(model, img, result)
    plt.imshow(mmcv.bgr2rgb(vis_result))
    plt.show()
