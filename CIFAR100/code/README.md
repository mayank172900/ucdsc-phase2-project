# Reaching Nirvana: Maximizing the Margin in Both Euclidean and Angular Spaces
**Abstract:** The classification loss functions used in deep neural network classifiers can be grouped into two categories based on maximizing the margin in either Euclidean or angular spaces. Euclidean distances between sample vectors are used during classification for the methods maximizing the margin in Euclidean spaces whereas the Cosine similarity distance is used during the testing stage for the methods maximizing the margin in the angular spaces. This paper introduces a novel classification loss that maximizes the margin in both the Euclidean and angular spaces at the same time. This way, the Euclidean and Cosine distances will produce similar and consistent results and complement each other, which will in turn improve the accuracies. The proposed loss function enforces the samples of classes to cluster around the centers that represent them. The centers approximating classes are chosen from the boundary of a hypersphere, and the pair-wise distances between class centers are always equivalent. This restriction corresponds to choosing centers from the vertices of a regular simplex inscribed in a hypersphere. There is not any hyperparameter that must be set by the user in the proposed loss function, therefore the use of the proposed method is extremely easy for classical classification problems. Moreover, since the class samples are compactly clustered around their corresponding means, the proposed classifier is also very suitable for open set recognition problems where test samples can come from the unknown classes that are not seen in the training phase. Experimental studies show that the proposed method achieves the state-of-the-art accuracies on open set recognition despite its simplicity.

![Nirvana](https://user-images.githubusercontent.com/67793643/217524225-82240880-27c7-4918-ab12-2e9b1235f701.png)
In the proposed method, class samples are enforced to lie closer to the class-specific centers representing them, and the class centers are located on the boundary of a hypersphere. All the distances between the class centers are equivalent, thus there is no need to tune any margin term. The class centers form the vertices of a regular simplex inscribed in a hypersphere. Therefore, to separate $C$ different classes, the dimensionality of the feature space must be at least $C-1$. The figure on the left shows the separation of 2 classes in 1-D space, the middle figure depicts the separation of 3 classes in 2-D space, and the figure on the right illustrates the separation of 4 classes in 3-D space. For all cases, the centers are chosen from a regular $C-$ simplex.

![PDAM](https://user-images.githubusercontent.com/67793643/217527332-b7962b96-d864-4a0a-bd81-fb8002d7e3d8.png)
The plug and play module that will be used for increasing feature dimension. It maps $d-$dimensional feature vectors onto a much higher $(C-1)-$ dimensional space.
# 1. Requirements
## Environments
Following packages are required for this repo.

    - python 3.8+
    - torch 1.9+
    - torchvision 0.10+
    - CUDA 10.2+
    - scikit-learn 0.24+
    - catalyst 21.10+
    - mlxtend 0.19+
 ## Datasets
 Cifar datasets and synthetic datasets are under the directory **'data'**. To train the network on face verification you need to download **MS1MV3** dataset. We have used a subset of this dataset including 12K individuals. We can provide this dataset for the interested users. Please send an email to hakan.cevikalp@gmail.com if you want this specific dataset.
# 2. Training & Evaluation
## Synthetic Experiments
- For synthetic experiments, simply run **'main_dsc_cifar_synthetic.py'** for our deafult loss function, **'main_dsc_cifar_synthetic_hinge.py'** for the hinge loss, and **'main_dsc_cifar_synthetic_softmax.py'** for softmax loss function. They will produce the figures used in Fig. 3 given in the paper.
## Open Set Recognition
- For open set recognition, simply run **'NirvanaOSR.py'** for to run experiments add datasets to "data" folder and choose dataset from mnist | svhn | cifar10 | cifar100 | tiny_imagenet one of them. Use "classifier32" networks for all experiments except tiny_imagenet and for that use resnet50. Use "NirvanaOpenset_loss" as loss function.
## Closed Set Recognition
- Just use **'main_cifar.py'** to train the network for Cifar100 dataset. The dataset is already under **data** folder. For other datasets, just downlaod the datasets and revise the main file to use the specific dataset.
## Experiments by Using DAM
- We have prepared some source files to show the effects of DAM. You can play with the source file to see the effects of parameters, e.g., expand factor (hyperspheer radius, u), number of classes, activation functions, etc. Simply run **'main_dsc_dam_module.py'**. It uses a simple LeNeT that yields 2-dimensional CNN features. DAM module increases the dimesion to any desired value. The defaul number of classes is 20 and they are chosen from Cifar-100 dataset. You can increase the number of classes. You can clearly see a big accuracy performance difference when different expand factors are used. Just set this value 24 and then 1000. You will see that setting u to 1000 yields significantly better accuracies compared to the one setting u to 24.
### Face Recognition Using DAM:
To train the network using DAM, just run **'NirvanaFaceDAM.py'**. It will fine tune from our previously trained deep simplex classifier using DAM. The pretrained model can found at the folder **./face_models/ResNet101_lr0.000001_Nirvana_Expand1000_DAM**. For evaluation, just set the **test_only** parameter to True at line 237. You can obtain the verifiction results given our paper.
### Face Recognition Using the Revised Network
To train the network using the revised network, just run **'NirvanaFaceRevisedNetwork.py'**. It will fine tune from our previously trained deep simplex classifier using revised network. The pretrained model can found at the folder **./face_models/ResNet101_lr0.000001_Nirvana_ResNet101_lr0.000100_Nirvana_Expand2000_Epoch5_NetworkRevised**. For evaluation, just set the **test_only** parameter to True at line 239. You can obtain the verifiction results given our paper.

# 3. Results
### The learned feature embeddings:
![embeddings](https://user-images.githubusercontent.com/67793643/217549694-0c4deabe-ed97-480f-8f29-2263534b0dda.png)
Learned feature representations of image samples: (a) the embeddings returned by the proposed method trained with the default loss function, (b) the embeddings returned by the proposed method trained with the hinge loss, (c) the embeddings returned by the proposed method trained with the softmax loss function.
### Semantically related feature embeddings:

![NirvanaHeatmap](https://user-images.githubusercontent.com/67793643/217550692-f3b65c68-9723-4fb5-ac1b-46d4fc3e32bf.png)
The distance matrix is computed by using the centers of the testing classes. The four classes that are not used in training are closer to their semantically related classes in the learned embedding space.

## Citation
```bibtex
@article{cevikalp2025,
  author    = {Hakan Cevikalp and Hasan Saribas and Bedirhan Uzun},
  title     = {Reaching nirvana: Maximizing the margin in both Euclidean and angular spaces for deep neural network classification},
  pages = {8178--8191},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume = {36},
  year      = {2025},
}





