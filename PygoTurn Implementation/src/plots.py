import matplotlib.pyplot as plt
import numpy as np

sequence = 'swan'
save_directory = "../our_results/plots/"

iou_delta_orig = np.load(save_directory + sequence + '/delta_iou_method_original.npy')
iou_delta_mean = np.load(save_directory + sequence + '/delta_iou_method_mean.npy')
iou_delta_iou = np.load(save_directory + sequence + '/delta_iou_method_iou.npy')
iou_fb_mean = np.load(save_directory + sequence + '/forward_backward_iou_method_mean.npy')
iou_fb_iou = np.load(save_directory + sequence + '/forward_backward_iou_method_iou.npy')

plt.figure()
plt.plot(iou_delta_orig)
plt.plot(iou_delta_mean)
plt.plot(iou_fb_mean)
plt.legend(['Original method', 'Delta method - mean', 'Forward-Backward method - mean'])
plt.grid()
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.title(sequence + ' sequence')
plt.savefig(save_directory + '/' + sequence + '_mean_methods.png')

plt.figure()
plt.plot(iou_delta_orig)
plt.plot(iou_delta_iou)
plt.plot(iou_fb_iou)
plt.title(sequence + ' sequence')
plt.legend(['Original method', 'Delta method - IoU', 'Forward-Backward method - IoU'])
plt.grid()
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.savefig(save_directory + '/' + sequence + '_iou_methods.png')

#plt.show()