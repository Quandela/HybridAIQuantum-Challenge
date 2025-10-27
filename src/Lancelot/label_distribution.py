import numpy as np
import matplotlib.pyplot as plt

# Verify the label distribution from a uniform output distribution

EFF_IMG_SIZE = 14
N = 7

# Generate a uniform output distribution
output = np.random.rand(10000, EFF_IMG_SIZE)
output = output / np.sum(output, axis=1)[:, None] * 6.8
print(output[0])

# Compute the label distribution

def output_to_float_label(output):
    return np.average(range(EFF_IMG_SIZE), weights=output)*9/(EFF_IMG_SIZE - 1)

def output_to_float_label_flat(output):
    return np.clip(np.sqrt(np.abs(np.average(range(EFF_IMG_SIZE), weights=output) - 6.5))*9 - 1, 0, 10)

print(output_to_float_label_flat(output[0]))

# Le théorème central limite nous impose de trouver une gaussienne mais la transformation 'flat' permet de l'applatir pour obtenir une distribution plus uniforme

labels = np.apply_along_axis(output_to_float_label, 1, output)
labels_exp = np.apply_along_axis(output_to_float_label_flat, 1, output)

# Verify the label distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(labels, bins=20, edgecolor='black')
axs[0].set_title('Label Distribution')
axs[0].set_xlabel('Label')
axs[0].set_ylabel('Frequency')

axs[1].hist(labels_exp, bins=20, edgecolor='black')
axs[1].set_title('Label Distribution (Flattened)')
axs[1].set_xlabel('Label')
axs[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()