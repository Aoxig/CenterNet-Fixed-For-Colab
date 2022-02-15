import numpy as np

inplanes = (64, 128, 256, 512)
planes = (256, 128, 64)
shortcut_num = min(len(inplanes) - 1, len(planes))
shortcut_cfg = (1, 2, 3)
print(inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num])
padding = (3 - 1) // 2
feat = [[1,2],[3,4]]
print(feat[:, 1:])