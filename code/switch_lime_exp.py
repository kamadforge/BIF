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


def patch_segmenter(img):
  assert img.shape[0] == 28 and img.shape[1] == 28
  patches = np.arange(49).reshape((7, 7))
  patches = np.repeat(np.repeat(patches, 4, axis=0), 4, axis=1)
  return patches

def get_batch_predict_fun(classifier, device):

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
  return batch_predict


def classifier_prep(ar):
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
  return classifier, device, test_data


def first_patch_lime_exp(ar):
  classifier, device, test_data = classifier_prep(ar)
  batch_predict = get_batch_predict_fun(classifier, device)

  test_image, test_label = test_data.__getitem__(0)
  test_label = test_label.item()
  test_image = np.reshape(test_image.numpy(), (28, 28))
  test_image = np.stack([test_image]*3, axis=2)
  # print(test_image)
  test_image = test_image.astype(np.float64)
  explainer = lime_image.LimeImageExplainer()

  # segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)


  explanation = explainer.explain_instance(test_image,
                                           batch_predict,  # classification function
                                           top_labels=2,
                                           hide_color=0,
                                           num_samples=1000,
                                           # segmentation_fn=segmenter,
                                           segmentation_fn=patch_segmenter)

  print(explanation.top_labels[0])
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                              hide_rest=False)
  plt.imsave('lime_img_boundary_1.png', mark_boundaries(temp, mask))
  plt.imsave('lime_img_boundary_1_no_boundary.png', label2rgb(mask, temp, bg_label=0))

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                              hide_rest=False)
  plt.imsave('lime_img_boundary_2.png', mark_boundaries(temp, mask))
  plt.imsave('lime_img_boundary_2_no_boundary.png', label2rgb(mask, temp, bg_label=0))

  weights = [0.0, 0.001, 0.01, 0.1, 0.3]
  for w in weights:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    temp, mask = explanation.get_image_and_mask(test_label, positive_only=True, num_features=10, hide_rest=False,
                                                min_weight=w)
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


def plot_patch_lime(ar, n_plots=10):
  """
  function to produce plots for the same 8 images used in the other methods.
  We produce plots both for the default LIME segmenter and for the segmentation in 7x7 grid shape
  """
  # get classifier
  classifier, device, test_data = classifier_prep(ar)
  batch_predict = get_batch_predict_fun(classifier, device)
  x_tst, y_tst = test_data.smp, test_data.tgt
  # get the 10 images used in the other experiments
  x_plt = np.concatenate([x_tst[:int(np.ceil(n_plots / 2))], x_tst[-int(np.floor(n_plots / 2)):]])
  y_plt = np.concatenate([y_tst[:int(np.ceil(n_plots / 2))], y_tst[-int(np.floor(n_plots / 2)):]])

  segment_lime_explanations = []
  patch_lime_explanations = []

  print(y_plt)
  # for each image get both lime explanations with the top k (1-5) patches shown
  for idx in range(x_plt.shape[0]):
    x = np.stack([np.reshape(x_plt[idx], (28, 28))]*3, axis=2).astype(np.float64)
    y = y_plt[idx]

    # get true lime explanation
    segment_imgs = explain(x, y, batch_predict, top_features_list=range(1, 6), patches=False)
    segment_lime_explanations.append(segment_imgs)

    # get patch lime explanation
    patch_imgs = explain(x, y, batch_predict, top_features_list=range(1, 6), patches=True)
    patch_lime_explanations.append(patch_imgs)

  segment_lime_explanations = np.concatenate(segment_lime_explanations, axis=1)
  patch_lime_explanations = np.concatenate(patch_lime_explanations, axis=1)

  plt.imsave('lime_segment_explanantions.png', segment_lime_explanations)
  plt.imsave('lime_patch_explanantions.png', patch_lime_explanations)

  np.save('lime_segment_explanantions.npy', segment_lime_explanations)
  np.save('lime_patch_explanantions.npy', patch_lime_explanations)

  segments_3 = segment_lime_explanations[2 * 28:3 * 28, :, :]
  segments_5 = segment_lime_explanations[4 * 28:5 * 28, :, :]

  patches_3 = patch_lime_explanations[2 * 28:3 * 28, :, :]
  patches_5 = patch_lime_explanations[4 * 28:5 * 28, :, :]

  plt.imsave('lime_segment_explanantions_3.png', segments_3)
  plt.imsave('lime_segment_explanantions_5.png', segments_5)
  plt.imsave('lime_patch_explanantions_3.png', patches_3)
  plt.imsave('lime_patch_explanantions_5.png', patches_5)


def explain(image, label, predict_fun, top_features_list, patches=False):
  explainer = lime_image.LimeImageExplainer()

  if patches:
    segmenter = patch_segmenter
  else:
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=5, ratio=0.2)

  explanation = explainer.explain_instance(image,
                                           predict_fun,  # classification function
                                           top_labels=2,
                                           hide_color=0,
                                           num_samples=1000,
                                           segmentation_fn=segmenter)
  masked_images = []
  for n_feat in top_features_list:
    _, mask = explanation.get_image_and_mask(label, num_features=n_feat)
    masked_images.append(make_masked_image(image, mask))

  masked_image_col = np.concatenate(masked_images, axis=0)
  return masked_image_col


  # label2rgb(mask, temp, bg_label=0)

def make_masked_image(image, mask):
  if len(image.shape) == 3:
    image = image[:, :, 0]
  # mask = mask.astype(np.bool)
  image = image / np.max(image)
  # print(image.shape, mask.shape, img_sel.shape)
  colored_sel = np.stack([image * mask, np.zeros_like(mask), (1 - image) * mask], axis=2)  # red on color, blue off color
  rest_img = np.stack([image * (1 - mask)]*3, axis=2)
  # print(np.min(rest_img), np.max(rest_img), np.min(colored_sel), np.max(colored_sel))
  # print(np.min(colored_sel + rest_img), np.max(colored_sel + rest_img))
  return colored_sel + rest_img

# def get_pos_only_top_k_image_and_mask(explanation_class, label, num_features, top_k=None):
#
#   segments = explanation_class.segments
#   image = explanation_class.image
#   exp = explanation_class.local_exp[label]
#   mask = np.zeros(segments.shape, segments.dtype)
#   temp = explanation_class.image.copy()
#
#   fs = [x[0] for x in exp if x[1] > 0][:num_features]
#
#   for f in fs:
#     temp[segments == f] = image[segments == f].copy()
#     mask[segments == f] = 1
#   return temp, mask, exp

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--classifier-epochs', type=int, default=10)
  # parser.add_argument('--switch-epochs', type=int, default=10)
  parser.add_argument('--classifier-lr', type=float, default=1e-4)
  # parser.add_argument('--switch-lr', type=float, default=3e-3)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--dataset', type=str, default='mnist')
  parser.add_argument('--label-a', type=int, default=3)
  parser.add_argument('--label-b', type=int, default=8)
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--big-classifier', action='store_true', default=True)

  return parser.parse_args()


def main():
  ar = parse_args()
  pt.manual_seed(ar.seed)
  np.random.seed(ar.seed)

  # first_patch_lime_exp(ar)
  plot_patch_lime(ar, n_plots=10)


if __name__ == '__main__':
  main()















