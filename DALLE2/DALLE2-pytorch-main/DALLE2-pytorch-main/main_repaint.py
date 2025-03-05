import torch
from dalle2_pytorch import Unet, Decoder, CLIP

# trained clip from step 1

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# 2 unets for the decoder (a la cascading DDPM)

unet = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 1, 1, 1)
).cuda()


# decoder, which contains the unet(s) and clip

decoder = Decoder(
    clip = clip,
    unet = (unet,),               # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
    image_sizes = (256,),         # resolutions, 256 for first unet, 512 for second. these must be unique and in ascending order (matches with the unets passed in)
    timesteps = 1000,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).cuda()

# mock images (get a lot of this)

images = torch.randn(4, 3, 256, 256).cuda()

# feed images into decoder, specifying which unet you want to train
# each unet can be trained separately, which is one of the benefits of the cascading DDPM scheme

loss = decoder(images, unet_number = 1)
loss.backward()

# do the above for many steps for both unets

mock_image_embed = torch.randn(1, 512).cuda()

# then to do inpainting

inpaint_image = torch.randn(1, 3, 256, 256).cuda()      # (batch, channels, height, width)
inpaint_mask = torch.ones(1, 256, 256).bool().cuda()    # (batch, height, width)

inpainted_images = decoder.sample(
    image_embed = mock_image_embed,
    inpaint_image = inpaint_image,    # just pass in the inpaint image
    inpaint_mask = inpaint_mask       # and the mask
)

inpainted_images.shape # (1, 3, 256, 256)

# save your image (in this example, of size 256x256)
import torch
from torchvision.utils import save_image


# 이미지 후처리: 텐서를 [0, 1] 범위로 정규화
# 만약 이미지가 -1과 1 사이에 있을 경우, (1 + image) / 2로 변환하여 [0, 1] 범위로 정규화
inpainted_images = (inpainted_images + 1) / 2.0

# 이미지를 CPU로 이동
inpainted_images = inpainted_images.cpu()

# save_image로 이미지 저장
save_image(inpainted_images, 'generated_inpaint_image.png')
print(save_image)

