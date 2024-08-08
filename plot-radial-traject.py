# Benjamin Yu July 2024
# Given 2 classes and their coordinates, will plot coordinates to visualize regional class distribution
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=5)

a_0 = np.loadtxt('single-point/prebuffer/trajectory_plot_7-23_1_0.0.txt')
b_0 = np.loadtxt('single-point/prebuffer/trajectory_plot_7-23_2_0.0.txt')
a_1 = np.loadtxt('single-point/prebuffer/trajectory_plot_7-23_1_9.0.txt')
b_1 = np.loadtxt('single-point/prebuffer/trajectory_plot_7-23_2_9.0.txt')

class_0 = np.concatenate((a_0, b_0), axis=0)
class_1 = np.concatenate((a_1, b_1), axis=0)
print(type(class_0[0]))

x_0, y_0 = class_0[:, 0], class_0[:, 1]
x_1, y_1 = class_1[:, 0], class_1[:, 1]

plt.figure(figsize=(12,9))

plt.scatter(x_0, y_0, color='blue', label='Class 0', s=1)
plt.scatter(x_1, y_1, color='red', label='Class 1', s=1)

plt.xlabel('X')
plt.ylabel('Y')
plt.title("Trajectories")
plt.legend()

plt.savefig('radial-trajectories.png', dpi = 1200)
plt.show()
