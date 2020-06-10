import numpy as np
import matplotlib.pyplot as plt

result_dir = "result/"

acc1 = np.load(result_dir + 'acc0r.npy')
loss1 = np.load(result_dir + 'loss0r.npy')*10
print(' 1: %.2f %.6f'%(100*acc1.max(), loss1.min()))

acc2s = np.load(result_dir + 'acc1s.npy')
loss2s = np.load(result_dir + 'loss1s.npy')*10
print('2s: %.2f %.6f'%(100*acc2s.max(), loss2s.min()))

acc2r = np.load(result_dir + 'acc1r.npy')
loss2r = np.load(result_dir + 'loss1r.npy')*10
print('2r: %.2f %.6f'%(100*acc2r.max(), loss2r.min()))

acc3s = np.load(result_dir + 'acc2s.npy')
loss3s = np.load(result_dir + 'loss2s.npy')*10
print('3s: %.2f %.6f'%(100*acc3s.max(), loss3s.min()))

acc3r = np.load(result_dir + 'acc2r.npy')
loss3r = np.load(result_dir + 'loss2r.npy')*10
print('3r: %.2f %.6f'%(100*acc3r.max(), loss3r.min()))

iter_ = np.arange(acc1.shape[0]) * 50

print(acc1.shape[0])

plt.figure()
p = plt.subplot(111)
p.plot(iter_, loss1, '-', label='0-layer + ReLU')
p.plot(iter_, loss2s, '-', label='1-layer + Sigmoid')
p.plot(iter_, loss2r, '-', label='1-layer + ReLU')
p.plot(iter_, loss3s, '-', label='2-layer + Sigmoid')
p.plot(iter_, loss3r, '-', label='2-layer + ReLU')
p.set_xlabel(r'# of Iterations')
p.set_ylabel(r'Loss')
p.legend(loc='upper right')
plt.tight_layout()
plt.savefig(result_dir + "loss.png")


plt.figure()
p = plt.subplot(111)
p.plot(iter_, acc1, '-', label='0-layer + ReLU')
p.plot(iter_, acc2s, '-', label='1-layer + Sigmoid')
p.plot(iter_, acc2r, '-', label='1-layer + ReLU')
p.plot(iter_, acc3s, '-', label='2-layer + Sigmoid')
p.plot(iter_, acc3r, '-', label='2-layer + ReLU')
p.set_xlabel(r'# of Iterations')
p.set_ylabel(r'Accuracy')
p.legend(loc='lower right')
plt.tight_layout()
plt.savefig(result_dir + "acc.png")

#  1: 23:24:44.414     Testing, total mean loss 0.019417, total acc 0.863300 - 23:24:33.131
# 2s: 20:20:39.807     Testing, total mean loss 0.003224, total acc 0.967700 - 20:18:21.597
# 2r: 20:48:01.448     Testing, total mean loss 0.002306, total acc 0.981300 - 20:45:16.709
#-2r: 20:38:47.940     Testing, total mean loss 0.002271, total acc 0.981500 - 20:35:59.910
# 3s: 00:38:10.865     Testing, total mean loss 0.001759, total acc 0.980098 - 00:33:01.622
# 3r: 21:24:04.253     Testing, total mean loss 0.001675, total acc 0.980588 - 21:19:28.262