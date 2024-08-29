import torch
import torch.nn as nn
from torchvision import models

model = models.vgg16(pretrained=True)

first_layer = model.features[0]
# Example input: a batch of images (batch_size=1, channels=3, height=224, width=224)
input_image = torch.randn(1, 3, 224, 224)

# Get the output of the first layer
output = first_layer(input_image)

print(output.shape)  # This will print the shape of the output tensor