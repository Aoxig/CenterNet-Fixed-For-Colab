inplanes = (64, 128, 256, 512)
planes = (256, 128, 64)
shortcut_num = min(len(inplanes) - 1, len(planes))
shortcut_cfg = (1, 2, 3)
print(inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num])
print([3]*2)
padding = (3 - 1) // 2
print(padding)
feat = [1, 2, 3, 4]
for i, (kernel_size, padding) in enumerate(zip([3,3,3,3], [1,1,1,1])):
    print(feat[-i - 2])