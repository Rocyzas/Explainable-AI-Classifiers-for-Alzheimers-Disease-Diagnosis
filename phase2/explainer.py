import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random

explainer = lime_image.LimeImageExplainer(random_state=42)

fig, ax = plt.subplots(5, 6, sharex='col', sharey='row')
fig.set_figwidth(20)
fig.set_figheight(16)
indecies = random.sample(range(sum(bad_predictions)), 5)

for j in range(5):
    explanation = explainer.explain_instance(X[bad_predictions][indecies[j]],
                                             model.predict,
                                             top_labels=5, hide_color=0, num_samples=1000,
                                             random_seed=42)
    ax[j,0].imshow(X[bad_predictions][indecies[j]])
    ax[j,0].set_title(le.classes_[y[bad_predictions][indecies[j]]])
    for i in range(5):
        temp, mask = explanation.get_image_and_mask(i, positive_only=True,
                                                    num_features=5, hide_rest=True)
        ax[j,i+1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ax[j,i+1].set_title('p({}) = {:.4f}'.format(le.classes_[i], y_pred_train[bad_predictions][indecies[j]][i]))
