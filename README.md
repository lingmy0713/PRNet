# PRNet

Mingyang Ling, Kan Chang, Mengyuan Huang, Hengxin Li, Shuping Dang, Baoxin Li, " PRNet: Pyramid Restoration Network for RAW Image Super-Resolution", Submitted to IEEE Trans. Comput. Imaging, 2023.

## Abstract
Typically, image super-resolution (SR) methods are applied to the standard RGB (sRGB) images produced by the image signal processing (ISP) pipeline of digital cameras.
However, due to error accumulation, low bit depth and the nonlinearity with scene radiance in sRGB images, performing SR on them is sub-optimal. To address this issue, a RAW image SR method called pyramid restoration network (PRNet) is proposed in this paper. Firstly, PRNet takes the low-resolution (LR) RAW image as input, and generates a rough estimation of the SR result in the linear color space. Afterwards, a pyramid refinement (PR) sub-network refines image details in the intermediate SR result and corrects its colors in a divide-and-conquer manner. To learn the appropriate colors for displaying, external guidance is extracted from the LR reference image in the sRGB color space, and then fed to the PR sub-network. To effectively incorporate the external guidance, the cross-layer correction module (CCM), which fully investigates the long-range interactions between two input features, is introduced in the PR sub-network. Moreover, as different frequency components decomposed from the same image are highly correlated, in the PR sub-network, the refined features from a lower layer are utilized to support the feature refinement in an upper layer. Extensive experiments presented in this paper demonstrate that the proposed method is capable of recovering fine details and small structures in images while producing vivid colors that align with the output of a specific camera ISP pipeline.

## Dependencies
* Python == 3.7.13
* Pytorch == 1.13.0
* torchvision == 0.14.0
* numpy == 1.21.5
* skimage == 0.19.3

## Test and Train 

The source code of PRNet will be available after the acceptance of this paper.
