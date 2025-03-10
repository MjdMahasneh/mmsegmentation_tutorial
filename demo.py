import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

# Check installations
print(torch.__version__, torch.cuda.is_available())
import mmseg
print(mmseg.__version__)

# Run Inference with MMSeg trained weight
config_file = './configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
video = mmcv.VideoReader('./video/video.mp4')
for frame in video:
   result = inference_model(model, frame)
   show_result_pyplot(model, frame, result, wait_time=0.1)




