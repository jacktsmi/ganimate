import pathlib
import os
import math
import numpy as np
from PIL import Image

# index = 0
# for path in pathlib.Path().iterdir():
#     p = str(path)
#     old_extension = path.suffix
#     directory = path.parent
#     if old_extension == '.py':
#         continue
#     if index < 10:
#         new_name = "00000" + str(index) + old_extension
#     elif index < 100:
#         new_name = "0000" + str(index) + old_extension
#     elif index < 1000:
#         new_name = "000" + str(index) + old_extension
#     else:
#         new_name = "00" + str(index) + old_extension
#     path.rename(pathlib.Path(directory, new_name))
#     index = index + 1

# index = 0
# for path in pathlib.Path().iterdir():
#     p = str(path)
#     print(p)
#     if path.suffix == '.py':
#         continue
#     if index < 10:
#         name = "00000" + str(index) + '.png'
#         new_name = "00000" + str(index) + '.jpg'
#     elif index < 100:
#         name = "0000" + str(index) + '.png'
#         new_name = "00000" + str(index) + '.jpg'
#     elif index < 1000:
#         name = "000" + str(index) + '.png'
#         new_name = "00000" + str(index) + '.jpg'
#     else:
#         name = "00" + str(index) + '.png'
#         new_name = "00000" + str(index) + '.jpg'
#     im = Image.open(name)
#     im = im.convert('RGB')
#     im1 = im.save(new_name)
#     index = index+1

#cur_path = pathlib.Path().parent.absolute()

index = 0
for path in pathlib.Path().iterdir():
    p = str(path)
    if path.is_file():
        if index < 10:
            directory = path.parent
            old_extension = path.suffix
            new_name = str(index) + old_extension
        elif index < 100:
            directory = path.parent
            old_extension = path.suffix
            new_name = str(index) + old_extension
        elif index < 1000:
            directory = path.parent
            old_extension = path.suffix
            new_name = str(index) + old_extension
        else:
            directory = path.parent
            old_extension = path.suffix
            new_name = str(index) + old_extension
    path.rename(pathlib.Path(directory, new_name))
    index = index + 1