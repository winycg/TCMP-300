import matplotlib.pyplot as plt
import os

directory_path = './checkpoint_mixup'


all_items = os.listdir(directory_path)


folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

fig_base_path = os.path.join(directory_path, 'figures')
if not os.path.exists(fig_base_path):
    os.makedirs(fig_base_path)
    
for folder in folders:
    log_path = os.path.join(directory_path, folder, folder+'.txt')

    fig_path = os.path.join(fig_base_path, folder+'.png')
    
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    accuracies = []
    for line in log_lines:
        if 'Test top-1 accuracy' in line:
            _, accuracy = line.strip().split('Test top-1 accuracy:')
            accuracies.append(float(accuracy))

    plt.figure()
    plt.plot(accuracies, label='Test top-1 accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test top-1 accuracy over epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(fig_path)
