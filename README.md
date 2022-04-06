# FaceSigns Demo
[ [Google Colab Demo] ](https://colab.research.google.com/drive/1Qzqw0x_R5Xt62stvJCCQDqP8Qw_HAhIp?usp=sharing) [[Project Webpage]](https://shehzeen.github.io/facesigns/) [[Paper]](https://arxiv.org/abs/2204.01960)

Inference demo for our paper [FaceSigns: Semi-Fragile Neural Watermarks for Media Authentication and Countering Deepfakes
](https://arxiv.org/abs/2204.01960)

<b>Abstract:</b> Deepfakes and manipulated media are becoming a prominent threat due to the recent advances in realistic image and video synthesis techniques. There have been several attempts at combating Deepfakes using machine learning classifiers. However, such classifiers do not generalize well to black-box image synthesis techniques and have been shown to be vulnerable to adversarial examples. To address these challenges, we introduce a deep learning based semi-fragile watermarking technique that allows media authentication by verifying an invisible secret message embedded in the image pixels. Instead of identifying and detecting fake media using visual artifacts, we propose to proactively embed a semi-fragile watermark into a real image so that we can prove its authenticity when needed. Our watermarking framework is designed to be fragile to facial manipulations or tampering while being robust to benign image-processing operations such as image compression, scaling, saturation, contrast adjustments etc. This allows images shared over the internet to retain the verifiable watermark as long as face-swapping or any other Deepfake modification technique is not applied. We demonstrate that FaceSigns can embed a 128 bit secret as an imperceptible image watermark that can be recovered with a high bit recovery accuracy at several compression levels, while being non-recoverable when unseen Deepfake manipulations are applied. For a set of unseen benign and Deepfake manipulations studied in our work, FaceSigns can reliably detect manipulated content with an AUC score of 0.996 which is significantly higher than prior image watermarking and steganography techniques.

## Try it on Google Colab

Try out FaceSigns on your browser using our [colab demo](https://colab.research.google.com/drive/1Qzqw0x_R5Xt62stvJCCQDqP8Qw_HAhIp?usp=sharing)

## Running the demo locally
To run the demo locally (using python3), install pytorch as per your cuda version. Additionally, install ``opencv2, pilgram`` using pip to evaluate FaceSigns against various image transformations.

## Citing our work

```
@article{facesigns2022,
  title={{FaceSigns: Semi-Fragile Neural Watermarks for Media Authentication and Countering Deepfakes}},
  author={Neekhara, Paarth and Hussain, Shehzeen and Zhang, Xinqiao and Huang, Ke and McAuley, Julian and Koushanfar, Farinaz},
  journal={arXiv:2204.01960},
  year={2022}
}

```
