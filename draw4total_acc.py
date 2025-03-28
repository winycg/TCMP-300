import matplotlib.pyplot as plt
import os
import numpy as np

directory_path = './checkpoint_mixup_new'


all_items = os.listdir(directory_path)


# model_names = ['swin_t', 'swin_s', 'swin_b', 'vit_b_16']
model_names = ['regnet_x_3_2gf', 'resnet50', 'convnext_tiny', 'densenet161', 'mobilenet_v3_large', 
               'shufflenet_v2_x1_5', 'efficientnet_b0']


# print_name = ['Swin-Tiny', 'Swin-Small', 'Swin-Base', 'ViT-Base-16']
# colors = ['blue', 'orange', 'green', 'red']
print_name = ['RegNet-X', 'ResNet-50', 'ConNeXt-Tiny', 'DenseNet-161', 'MobileNetV3-Large', 
               'ShuffeNetV2', 'EffcientNet-BO']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan']



plt.figure()

for i, model in enumerate(model_names):
    log_path = os.path.join(directory_path, 'train_baseline_arch_'+model+'_seed0', 'train_baseline_arch_'+model+'_seed0.txt')

    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    accuracies = []
    for line in log_lines:
        if 'Test top-1 accuracy' in line:
            _, accuracy = line.strip().split('Test top-1 accuracy:')
            accuracies.append(float(accuracy))

    plt.plot(accuracies, color=colors[i], label=print_name[i])
    
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
# plt.title('Test top-1 accuracy over epochs')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(os.path.join(directory_path, 'cnn_acc.pdf'))
