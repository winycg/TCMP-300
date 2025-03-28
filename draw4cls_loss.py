import matplotlib.pyplot as plt


log_file = './checkpoint/train_baseline_arch_convnext_tiny_seed0/train_baseline_arch_convnext_tiny_seed0.txt'
with open(log_file, 'r') as f:
    lines = f.readlines()


test_losses = []
for line in lines:
    if 'Test_loss_cls' in line:
        values = line.split('Test_loss_cls:')[1].strip().split(' ')
        test_losses.append(float(values[0]))


epochs = len(test_losses)


plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('figures/loss.png')
