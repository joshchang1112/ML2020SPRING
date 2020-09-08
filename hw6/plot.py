import matplotlib.pyplot as plt
import numpy as np
index = np.arange(3)
top1 = 0.3672
top2 = 0.0940
top3 = 0.0808
plt.bar(index, (top1, top2, top3), 0.5)
plt.title('Adversarial Image')
plt.xticks(index ,("buckler(787)","whirligig(476)", "vestment(887)"))
plt.ylim(0.0, 1.0)
plt.show()
