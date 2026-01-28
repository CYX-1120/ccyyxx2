from SpliceMix import SpliceMix, CAMGuidedMixer
import torch

mixer = CAMGuidedMixer()
inputs_group = torch.rand(4, 3, 224, 224)
sal_maps = torch.rand(4, 224, 224)

sal_maps[0, 0:112, 0:112] = 0.8
sal_maps[0, 0:112, 112:224] = 0.2
sal_maps[0, 112:224, 0:112] = 0.4
sal_maps[0, 112:224, 112:224] = 0.2
sal_maps[1, 0:112, 112:224] = 0.9
sal_maps[2, 112:224, 0:112] = 0.85
sal_maps[3, 112:224, 112:224] = 0.9

mixed, mix_info = mixer.apply_cam_guided_mix(
    inputs_group, sal_maps, 
    g_row=2, g_col=2,
    current_epoch=40, total_epochs=80
)

print(f'混合后形状: {mixed.shape}')
print(f'统计: {mix_info["stats"]}')
stats = mix_info['stats']
print(f'anchor={stats["keep_anchor"]}, donor={stats["use_donor"]}, cam_guided={stats["cam_guided"]}')
print('测试通过！')
