import dataset
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
 ])
train_A = dataset.ImageData('dataA', 'train', transform=transf)
test_A = dataset.ImageData('dataA', 'test', transform=transf)

train_loader = DataLoader(train_A, batch_size=1, shuffle=True)
test_loader = DataLoader(test_A, batch_size=1, shuffle=False)

fig, axes = plt.subplots(2, 2)
axes = np.reshape(axes, (4, ))
for i in range(4):
    example = next(iter(train_loader))[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    example = std * example + mean
    axes[i].imshow(example)
    axes[i].axis('off')
plt.show()