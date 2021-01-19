"""
Test learning feature importance under DP and non-DP models
"""

__author__ = 'anon_m'

import argparse
import numpy as np

import torch.nn.functional as nnf
# from torch.nn.parameter import Parameter
# import sys
import torch as pt
from switch_model_wrapper import BinarizedMnistNet,  BigBinarizedMnistNet
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
from mnist_utils import train_classifier, BinarizedMnistDataset, DataLoader
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, rgb2gray, label2rgb


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--classifier-epochs', type=int, default=10)
  parser.add_argument('--switch-epochs', type=int, default=10)
  parser.add_argument('--classifier-lr', type=float, default=1e-4)
  parser.add_argument('--switch-lr', type=float, default=3e-3)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--dataset', type=str, default='mnist')
  parser.add_argument('--label-a', type=int, default=3)
  parser.add_argument('--label-b', type=int, default=8)
  parser.add_argument('--select-k', type=str, default='1,2,3,4,5')
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--big-classifier', action='store_true', default=True)
  parser.add_argument('--big-selector', action='store_true', default=True)
  # parser.add_argument("--freeze-classifier", default=True)
  # parser.add_argument("--patch-selection", default=True)

  return parser.parse_args()


def do_featimp_exp(ar):
  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_data = BinarizedMnistDataset(train=True, label_a=ar.label_a, label_b=ar.label_b,
                                     data_path='../data', tgt_type=np.float32)
  test_data = BinarizedMnistDataset(train=False, label_a=ar.label_a, label_b=ar.label_b,
                                    data_path='../data', tgt_type=np.float32, shuffle=False)

  train_loader = DataLoader(train_data, batch_size=ar.batch_size, shuffle=True, **kwargs)
  test_loader = DataLoader(test_data, batch_size=ar.test_batch_size, shuffle=False, **kwargs)
  # unpack data

  classifier = BigBinarizedMnistNet().to(device) if ar.big_classifier else BinarizedMnistNet().to(device)

  # classifier.load_state_dict(pt.load(f'models/{ar.dataset}_nn_ep4.pt'))
  train_classifier(classifier, train_loader, test_loader, ar.classifier_epochs, ar.classifier_lr, device)
  print('Finished Training Classifier')

  def batch_predict(images):

    classifier.eval()
    # batch = pt.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    if isinstance(images, (list, tuple)):
      batch = np.stack(images, dim=0)
    else:
      batch = images

    if len(batch.shape) == 4:
      assert batch.shape[3] == 3
      batch = batch[:, :, :, 0]
    if batch.dtype == np.float64:
      batch = batch.astype(np.float32)
    batch = pt.tensor(batch, device=device)
    batch = pt.reshape(batch, (-1, 784))

    binary_logits = classifier(batch)
    binary_probs = pt.sigmoid(binary_logits)
    vector_probs = pt.stack([binary_probs, 1 - binary_probs], dim=1)
    return vector_probs.detach().cpu().numpy()

  test_image, test_label = test_data.__getitem__(0)
  test_label = test_label.item()
  test_image = np.reshape(test_image.numpy(), (28, 28))
  test_image = np.stack([test_image]*3, axis=2)
  # print(test_image)
  test_image = test_image.astype(np.float64)
  explainer = lime_image.LimeImageExplainer()
  segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

  explanation = explainer.explain_instance(test_image,
                                           batch_predict,  # classification function
                                           top_labels=2,
                                           hide_color=0,
                                           num_samples=1000,
                                           segmentation_fn=segmenter)

  print(explanation.top_labels[0])
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                              hide_rest=False)
  img_boundry1 = mark_boundaries(temp, mask)
  plt.imsave('lime_img_boundary_1.png', img_boundry1)

  plt.imsave('lime_img_boundary_1_no_boundary.png', label2rgb(mask, temp, bg_label=0))

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                              hide_rest=False)
  img_boundry2 = mark_boundaries(temp, mask)
  plt.imsave('lime_img_boundary_2.png', img_boundry2)

  plt.imsave('lime_img_boundary_2_no_boundary.png', label2rgb(mask, temp, bg_label=0))

  weights = [0.0, 0.001, 0.01, 0.1, 0.3]
  for w in weights:
    temp, mask = explanation.get_image_and_mask(test_label, positive_only=True, num_features=10, hide_rest=False,
                                                min_weight=w)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.set_title(f'Positive Regions for {test_label}')
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')

    temp, mask = explanation.get_image_and_mask(test_label, positive_only=False, negative_only=True,
                                                num_features=10, hide_rest=False, min_weight=w)
    ax2.set_title(f'Negative Regions for {test_label}')
    ax2.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')

    temp, mask = explanation.get_image_and_mask(test_label, positive_only=False, num_features=10, hide_rest=False,
                                                min_weight=w)
    ax3.set_title(f'Positive/Negative Regions for {test_label}')
    ax3.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')

    plt.savefig(f'lime_regions_min_weight_{w}.png')


def main():
  ar = parse_args()
  pt.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  do_featimp_exp(ar)


if __name__ == '__main__':
  main()















