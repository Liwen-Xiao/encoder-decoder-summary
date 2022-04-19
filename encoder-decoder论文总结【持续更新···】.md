（注：Github 上的 markdown 是不支持 LaTex 语法的，所以在 Github 上这篇文章里的很多公式都渲染不出来，但如果想看渲染好的公式的话，请将本仓库下载到本地，使用 VScode + Markdown All in One 插件来阅读本文。）

---

# 序言
这篇文章是作者在学习encoder-decoder的网络结构的过程中，阅读相关文献，对文献的总结、归纳和结合自己的理解完成的一篇文章。若有错误敬请指出，本人不胜感激。

这篇文章中涉及到的文献：

网络结构：

> + **FCN**  *(CVPR 2015 best paper)*
> + **U_Net**  *(MICCAI 2015)*
> + **DeconvNet** *(ICCV 2015)*
> + **Stacked Hourglass Networks** *(ECCV 2016)*
> + **Associative Embedding: End_to_End Learning for Joint Detection and Grouping** *(NIPS 2017)*
> + **SegNet** *(PAMI 2017)*
> + **FPN**  *(CVPR 2017)*
> + **PANet** *(CVPR 2018)*
> + **RefineNet** *(CVPR 2017)*
> + **Deeplab_v3** *(CVPR 2017)*
> + **DeepLab_v3+** *(ECCV 2018)*
> + **NAS_FPN** *(CVPR 2019)*
> + **DUpsample** *(CVPR 2019)*
> + **the devil is in the decoder** *(IJCV 2019)*
> + **EffecientNet** *(ICML 2019)*
> + **BiFPN** *(CVPR 2020)*
> + **NRD** *(NeurIPS 2021)*
> + **EfficientFCN** *(TPAMI 2021)*


对空洞卷积的改进：
> + **Understanding Convolution for Semantic Segmentation** *(WACV 2018)*
> + **Dilated Residual Networks** *(CVPR 2017)*
> + **Smoothed Dilated Convolutions for Improved Dense Prediction** *KDD 2018*
> + **Deeplab_v3** *(CVPR 2017)*
> + **ESPNet: Efficient Spatial Pyramid of DilatedConvolutions for Semantic Segmentation** *(ECCV 2018)*
> + **Tree_structured Kronecker Convolutional Networks for Semantic Segmentation** *(ICME 2019)*
> + **Concentrated_Comprehensive Convolutionsfor lightweight semantic segmentation** *(2018)*)

___

# 目录
## Chapter 1 网络结构介绍
> ### Section 1 朴素的 Residual_like 网络
>>> + FCN
>>> + U_Net
>>> + DeconvNet
>>> + SegNet
> ### Section 2 进阶的结构
>> #### Branch 1 FPN_base 结构
>>> + FPN
>>> + PANet
>>> + Deeplab v3
>>> + Deeplab v3+
>>> + BiFPN
>>> + EffecientFPN
>> #### Branch 2 decoder_focus 结构
>>> + DUpsample
>>> + bilinear additive upsampling（+conv）
>>> + NRD
>> #### Branch 3 U_Net_Pro 结构
>>> + RefineNet
>> #### others
>>> + Stacked Hourglass Networks

## Chapter 2 重要部件介绍
> ### Section 1 转置卷积
> ### Section 2 对四种 decoder 的考察
>>> + Transposed Convolution
>>> + Decomposed Transposed Convolution
>>> + Depth_To_Space
>>> + bilinear
> ### Section 3 对空洞卷积的改进
> ### Section 4 Fast Normalized Fusion

## Chapter 3 自己的思考

_ _ _

# chapter 1 网络结构介绍
## Section 1 朴素的 Residual_like 网络

包括的网络结构有：
> + **FCN**  *(CVPR 2015 best paper)*
> + **U_Net**  *(MICCAI 2015)*
> + **DeconvNet** *(ICCV 2015)*
> + **SegNet** *(PAMI 2017)*

**说明**：这类网络思路简单，受到 ResNet 的启发，将网络结构设计为 Residual_like 的结构，在最终的输出中，既有深层网络特征，又融合了浅层网络特征。

我们首先来看看这些网络的网络结构，再来分析其动机和效果。

**网络结构为：**

### FCN

<div align = center> <img src="pictures/FCN_.png "/></div>

FCN中下采样使用的是：maxpooling  
FCN中上采样采用的是：转置卷积
FCN中特征融合的操作是：add

其中：
①直接将 16 * 16 的特征图上采样成 568 * 568 的原始尺寸的结构被称为 **FCN_32s**：

 

<div align = center> <img src="pictures/FCN_32s.png "/></div>

 

②先将 $\frac{1}{32}$ 的特征图和  $\frac{1}{16}$ 的特征图进行融合，再对融合得到的 $\frac{1}{16}$ 的特征图进行16倍的上采样，得到原始尺寸的图，这样的结构被称为：**FCN_16s**：

 
<div align = center> <img src="pictures/FCN_16s.png "/></div>

 

②先将 $ \frac{1}{32} $ 的特征图和 $\frac{1}{16}$ 的特征图进行融合，再对融合得到的 $\frac{1}{16}$ 的特征图与 $\frac{1}{8}$ 的特征图进行融合，再对融合得到的 $\frac{1}{8}$ 的特征图进行8倍的上采样得到原始尺寸的图，这样的结构被称为：**FCN_8s**：

 
<div align = center> <img src="pictures/FCN_8s.png "/></div>

 
### U_Net

 
<div align = center> <img src="pictures/U_Net_.png "/></div>

 

UNet中下采样使用的是：maxpooling  
UNet中上采样采用的是：转置卷积
UNet中特征融合的操作是：catch

>**总结1**：对比FCN和UNet的结构，我认为可以将UNet看作是FCN的“完备版”：UNet是将每一层encoder的特征都与后来decoder的特征进行融合，即：UNet就是FCN_2s.

>*挖坑：在FCN的paper中，作者提到了：FCN_4s、FCN_2s的效果并没有FCN_8s的效果好，也就是说UNet中的decoder结构是存在冗余的，而UNet效果是比FCN好的，所以可以去考察为什么UNet的效果比FCN好，我觉得可能的点有：UNet paper中提出的data augmentation、更大的数据集等*

### Deconvnet

 
<div align = center> <img src="pictures/Deconvnet_.png "/></div>

 

Deconvnet中下采样使用的是：maxpooling  
Deconvnet中上采样采用的是：转置卷积
Deconvnet中特征融合的操作是：无特征融合

在Deconvnet的结构中，没有特征融合操作。值得注意的是，在代码中，decoder中的卷积操作是使用的转置卷积（Chapter 2 中有证明：不带上采样的转置卷积与标准卷积没有区别）

### SegNet

 
<div align = center> <img src="pictures/segnet_.png "/></div>

 
SegNet中下采样使用的是：带位置信息的maxpooling  
SegNet中上采样采用的是：带位置信息的unmaxpooling
SegNet中特征融合的操作是：无特征融合

在 SegNet 的结构中，没有特征融合操作，之所以称其为Residual-like的结构，是因为其decoder的unpooling中使用的位置信息是在encoder的maxpooling中得到的。
SegNet和Deconvnet的结构的区别：
①每层特征图的尺寸不同：DeconvNet 中有全连接层，而 SegNet 全为卷积层
②在代码中，SegNet中 decoder 的上采样使用的带位置信息的 填充0的上采样，while Deconvnet中采用的是转置卷积上采样

>**总结2**：Deconvnet和SegNet中的**带坐标信息的maxpooling**和**带坐标信息的unpooling**（这两个上下采样操作的具体操作会在 chapter 2 中详细介绍）与 FCN 中的**maxpooling**和**转置卷积**的区别:

>Deconvnet 和 SegNet 在 Unpooling 时用 index 信息，直接将数据放回对应位置，后面再接Conv训练学习。这个上采样不需要训练学习(只是占用了一些存储空间)。反观FCN则是用transposed convolution策略，即将feature 反卷积后得到upsampling，这一过程需要学习，同时将encoder阶段对应的feature做通道降维，使得通道维度和upsampling相同，这样就能做像素相加得到最终的decoder输出.

## Section 2 进阶的结构
后来的encoder_decoder结构走向了三个发展方向：
①encoder采用FPN，研究很好地融合各尺度特征的decoder——FPN_base 
②encoder采用CNN、ResNet等经典的backbone，在encoder中引入Residual_like结构（像UNet那样），主要贡献和研究点在decoder的设计上（有可学习的decoder，也有融合各种decoder特征的decoder）
③在UNet的结构上继续优化（以RefineNet为代表）
### Branch 1  FPN_base 结构

包括的网络结构有：

> + FPN
> + PANet
> + Deeplab v3
> + Deeplab v3+
> + BiFPN
> + NAS_FPN
> + EffecientFPN


#### FPN

<div align = center> <img src="pictures/FPN_1.png "/></div>
 

(d)中的结构为 FPN 的结构

另外：
(a) 中的结构中特征在每一个维度分别计算，效率低下  
(b) FPN提出前主流的单尺度目标检测的网络，也就是普通的CNN  
(c) 利用 CNN 的每一层特征做直接的预测

FPN 的结构其实与 U_Net 的结构如出一辙，不同的地方在于输出侧：FPN直接利用上采样+特征融合后的每一层特征进行预测，而 U_Net 仅使用最后一层特征进行预测，如下图：（上面的网络是 U_Net 的结构，下面的网络是 FPN 的结构）

 <div align = center> <img src="pictures/FPN_2.png "/></div>


 

#### PANet

 <div align = center> <img src="pictures/PANet_1.png "/></div>
 

结构说明：
(a) **FPN**  
(b) **Bottom_up path augmentation**: 再用一个 Residual_like 的结构来一次下采样  
(c) **Adaptive Feature Pooling**: 将每一层的特征进行池化，得到尺寸相同的特征图，以便于接下来融合各层特征  
(d) **Box branch**：将融合了的2维特征拉成一维的，进行全连接，用于分类、检测任务  
(e) **Fully_connected fusion**: 对融合了的2维特征进行conv和residual_like连接，用于稠密预测任务

下图为(c)+(d)的结构：

 <div align = center> <img src="pictures/PANet_2.png "/></div>
 

下图为(e)的结构：

 <div align = center> <img src="pictures/PANet_3.png "/></div>
 

*（挖坑：(c) 中特征融合的时候，各层特征是否有权重呢，这个权重是否可以通过学习来得到呢）*

#### Deeplab v3

<div align = center> <img src="pictures/deeplabv3_3.png "/></div>


Deeplab v3 的主要创新点和贡献是：提出了 **ASPP（Atrous Spatial Pyramid Pooling）**

ASPP ：使用不同 rate 的空洞卷积使用不同的感受野大小提取特征，然后将这些通过不同感受野的特征concat起来组成最终特征。这样的操作可以使得提取的特征具有高分辨率的粗粒度与细粒度的特征。

ASPP 就是 Atrous convolution 和 SPP（Spatial Pyramid Pooling）的组合。

##### Atrous convolution

<div align = center> <img src="pictures/kdjj.png "/></div>

stride = 1 的空洞卷积的效果：在不减少特征图尺寸的情况下增大感受野，如下图为空洞卷积的朴素的应用：

<div align = center> <img src="pictures/deeplabv3_1.png "/></div>

##### SPP（Spatial Pyramid Pooling）

 <div align = center> <img src="pictures/deeplabv3_2.png "/></div>

将原本串联的特征金字塔的特征变成并联的，在并联结构中使用的是不同大小卷积核的卷积，并将它们concat起来（必要时回对小的特征进行上采样，上采样操作一般为双线性插值），这样最终得到的特征就具有各种感受野（粗细粒度）的特征且具有较高分辨率。

#### Deeplab v3+

 <div align = center> <img src="pictures/Dlabv3p_1.png "/></div>
 

 <div align = center> <img src="pictures/Dlabv3p_2.png "/></div>
 

以上两张图都是对 Deeplab v3+ 结构的描述：ASPP 与 encoder_decoder 结构的组合—— Deeplab v3 作为 encoder，decoder为设计的一个简单的结构（在我看来这个 decoder 就是一个阉割版的 U_Net）

同时，空洞卷积使得参数量和计算量很大（因为空洞卷积总是在分辨率较高的特征图上进行操作，且不会减小特征图尺寸），作者提出了一下结构以减少参数量和计算量：  
① Depthwise separable convolution（深度可分离卷积）
② Modified Aligned Xception（改进后的 Aligned Xception）

首先，我们先细节地描述一下整体的网络结构，再来考察一下 Depthwise separable convolution 和 Modified Aligned Xception 的有效性： 

##### 结构

DeepLabv3的编码器特征通常在输出步幅=16的情况下进行计算。特征被双线性上采样16倍，这可以被认为是原始的解码器模块。但是，这种原始的解码器模块可能无法成功恢复目标分割的详细信息。因此，我们提出了一个简单而有效的解码器模块，如下图所示。首先对编码器特征进行双线性插值上采样，放大倍数为4，然后将其与来自网络主干的具有相同空间分辨率的低级特征连接起来。我们在低层次特征上应用了另外的1×1卷积以减少通道数量，因为相应的低层次特征通常包含大量通道（例如256或512），这可能超过了丰富编码器功能的重要性（在我们的模型中只有256个通道），使训练变得更加困难。串联后，我们应用了3×3的卷积来细化特征，然后再进行简单的双线性上采样，其倍数为4。使用编码器模块的输出步幅=16可以在速度和精度之间取得最佳平衡。当将输出步幅=8用于编码器模块时，性能会略有提高，但代价是额外的计算复杂性。

##### Depthwise separable convolution（深度可分离卷积）

深度可分离卷积的操作步骤是：
第一步：对每一个通道的特征进行卷积，得到分分别只含有一维通道信息的特征集合
第二步：对得到的特征集合进行 1*1 的卷积，得到融合了原始特征所有通道维度信息的特征
如下图所示：

<div align = center> <img src="pictures/DSC_1.png "/></div>

<div align = center> <img src="pictures/DSC_2.png "/></div>

while，常规的卷积操作如图所示：

<div align = center> <img src="pictures/DSC_3.png "/></div>

我们来比较两种卷积的参数量，首先，我们考察常规卷积的参数量（以上面的图中的参数为例）：

$$ N_{standard} = 4 \times 3 \times 3 \times 3 = 108 $$

相应的深度可分离卷积的参数量为：

$$ N_{depthwise} = 3 \times 3 \times 3 = 27 $$
$$ N_{pointwise} = 1 \times 1 \times 3 \times 4 = 12 $$
$$ N_{separable} = N_{depthwise} + N_{pointwise} = 39 $$

所以我们可以看出，深度可分离卷积是可以很好地减少参数量的。

在原文中，作者使用的是深度可分离空洞卷积，如下图示：

<div align = center> <img src="pictures/DSC_4.png "/></div>

##### Modified Aligned Xception

Inception的核心思想是将channel分成若干个不同感受野大小的通道，除了能获得不同的感受野，Inception还能大幅的降低参数数量。我们看下图中一个简单版本的Inception模型：

对于一个输入的Feature Map，首先通过三组 1 * 1 卷积得到三组Feature Map，它和先使用一组 1 * 1 卷积得到Feature Map，再将这组Feature Map分成三组是完全等价的（再下一张图）。假设图1中 1 * 1 卷积核的个数都是 $ k_1 $ ， 3 * 3 的卷积核的个数都是 $ k_2 $ ，输入Feature Map的通道数为 $ m $ ，那么这个简单版本的参数个数为:

$$ m \times k_1 + 3 \times 3 \times 3 \times \frac{k_1}{3} \times \frac{k_2}{3} = m \times k_1 + 3 \times k_1 \times k_2 $$

对比相同通道数，但是没有分组的普通卷积，普通卷积的参数数量为：

$$ m \times k_1 + 3 \times 3 \times k_1 \times k_2 $$

参数数量约为Inception的三倍.

<div align = center> <img src="pictures/Xception_1.png "/></div>

<div align = center> <img src="pictures/Xception_2.png "/></div>
 

如果Inception是将 3 * 3 卷积分成3组，那么考虑一种极端的情况，我们如果将Inception的 1 * 1 得到的 $ k_1 $ 个通道的Feature Map完全分开呢？如下图所示。也就是使用 $ k_1 $ 个不同的卷积分别在每个通道上进行卷积，它的参数数量是：

$$ m \times k_1 + 3 \times 3 \times k_1 $$

更多时候我们希望两组卷积的输出Feature Map相同，这里我们将Inception的 1 * 1 卷积的通道数设为 $ k_2 $ ，即参数数量为:

$$ m \times k_1 + 3 \times 3 \times k_2 $$

它的参数数量是普通卷积的 $ \frac{1}{k_1} $ ，我们把这种形式的Inception叫做Extreme Inception。如下图所示： 

 <div align = center> <img src="pictures/Xception_3.png "/></div>
 

多层的 Extreme Inception 被称作 Xception。

Inception、Xception 本质上是用 **通道信息融合程度** 来换 **参数量、计算成本**，因为最后 concat 得到的特征图中不同片段之间信息是独立的，但是，从一个 Xception 单元来看，他们是独立的，但是在多个 Xception 串联起来看，每个 concat 后的特征图都会输入到 下一个 Xception 单元的 $ 1 \times 1 conv $ 里，信息再次被融合。所以我们可以认为：常规卷积对信息的融合 **冗余** 了，就像一直在搅拌一杯充分溶解的糖水，吃力不讨好。

Xception模型在ImageNet上显示了具有潜力的图像分类结果，并且运算速度很快。MSRA团队修改了Xception模型（称为Aligned Xception），并进一步提高了目标检测任务的性能。受这些发现的启发，Deeplab v3+ 朝着相同的方向努力以使Xception模型适应语义图像分割的任务。特别是，Deeplab v3+ 在MSRA修改的基础上进行了一些其他更改，即（1）更深的Xception，不同之处在于Deeplab v3+ 不修改entry flow网络结构以实现快速计算和存储效率，（2）全部max pooling操作被有步长的深度可分离卷积替代，这使 Deeplab v3+ 能够应用空洞可分离卷积以任意分辨率提取特征图（另一种选择是将空洞算法扩展到最大池化操作），以及（3）额外批处理，每进行3×3深度卷积后，就添加归一化和ReLU激活，类似于MobileNet设计。改进后的 Xception 结构如下图所示：

<div align = center> <img src="pictures/Xception_4.png "/></div>

> 总结：深度可分离卷积在我看来和Xception几乎是等价的，区别之一就是 Xception 先计算Pointwise卷积再计算Depthwise的卷积，while 先计算深度可分离卷积再计算 Depthwise的卷积。

Xception 的效果如下所示（此处 Xception 和 ResNet 参数量相同）：

<div align = center> <img src="pictures/Xception_5.png "/></div>

可以看出 Xception 相较于 ResNet 是有优势的。

#### BiFPN

<div align = center> <img src="pictures/BiFPN_1.png "/></div>

BiFPN 的 backbone 直接使用的是 EffecientNet 的结构，其贡献为创造了一个 decoder，其 decoder 结构如下图 (d) 所示：

<div align = center> <img src="pictures/BiFPN_2.png "/></div>

在上图中，(a) 结构为基础的 FPN 的结构。(b) 结构为 PANet 的 decoder 结构。(c) 结构为 NAS_FPN 的 decoder 结构。

BiFPN 的效果为：
① 更好的效果（ $ mIOU $ 比 Deeplab v3+ ( Xception ) 高 $ 1.7\% $ )
② 更低的计算量 （计算量仅为 Deeplab v3+ ( Xception ) 的 $ \frac{1}{10} $

>在我看来，更低的计算量不是 BiFPN 的贡献，而是 EffecientNet 这个 backbone 的贡献，虽然但是，BiFPN 效果还是比 FPN 要好一些的。

BiFPN 的效果如下图所示：

 <div align = center> <img src="pictures/BiFPN_3.png "/></div>

<div align = center> <img src="pictures/BiFPN_4.png "/></div>
 

##### EffecientNet

<div align = center> <img src="pictures/EffecientNet_1.png "/></div>

EffecientNet的作者通过尝试和求解规划问题（NAS技术）来确定使得 ①计算量和参数量在一定范围内 ②效果最好  的网络结构参数：**depth**、**number of channals (width)**、**resolution**。

对我们来说，不用去深究和 follow 网络结构的求解过程，因为搜索的计算量实在太大（连 Google 自己都觉得大），所以我们以后把 EffecientNet 当作 backbone 来使用就行了。

#### EffecientFPN

下图中 (d) 中的结构即为 EfficientFPN 的结构：

<div align = center> <img src="pictures/EffecienFPN_1.png "/></div>

下图为更细节的 EfficientFPN 的结构：

<div align = center> <img src="pictures/EffecienFPN_2.png "/></div>

EffecientFPN 结构整体上来说，核心思想还是 **融合低层次细粒度的信息和高层次粗粒度的信息** ：作者使用尺寸较小的特征图$(\frac{1}{32})$来生成 codewords , 在 codewords 中有较为全局的特征，但是细节不够。使用尺寸较大的特征图$(\frac{1}{8})$来提取细粒度的信息；最后将两种层次的信息进行融合。

网络具体的细节有：
① codewords 中使用的小尺寸的特征图$(\frac{1}{32})$来自三种尺寸的特征图（这三种尺寸的特征图来源于 FPN ）的融合：$\frac{1}{16}$的特征图进行 $\times \frac{1}{2}$ 的 pooling，$\frac{1}{8}$的特征图进行 $\times \frac{1}{4}$ 的 pooling，最后将三个来源的特征图 concat 起来。
② codewords assembly coefficients 中使用的大尺寸的特征图$(\frac{1}{8})$同样也是来自三种尺寸的特征图（这三种尺寸的特征图来源于 FPN ）的融合：$\frac{1}{32}$的特征图进行 $\times 4$ 的 bilinear，$\frac{1}{16}$ 的特征图进行 $\times 2$ 的 bilinear，最后将三个来源的特征图 concat 起来。
③ 将 codewords 和 codewords assembly coefficients 中的粗粒度和细粒度的特征结合起来

>总结：（完全是个人看法）在我看来，这个结构和 U_Net 的结构的本质的不同在于 **粗粒度信息和细粒度信息出现在网络流程中的位置不同** ：在 U_Net 的结构中，粗粒度的信息出现在网络流程的中间，细粒度信息出现在网络结构的首尾；while，EffecientFPN 的结构中，粗粒度信息和细粒度的信息是平等的：在 codewords 和 codewords assembly coefficients 中都先是融合了两种信息，再做后续操作。**在 U_Net 中，我们可以认为粗粒度信息和细粒度信息都没有被充分地利用，因为： 1）两处粗粒度信息 concat 时，中间的操作不多，导致差异性不大，信息未利用充分，而细粒度的信息相隔较远，差异较大，在这个层面利用充分。 2） 但是 通过 Residual 过来的细粒度信息后没有经过太多操作就输出了，导致信息处理不到位，信息未利用充分**。从这个层面上来讲，EffecientFPN 消除了两种信息在网络中的不对称的状况，使得网络效果更好。

EffecientFPN 的效果如下图所示：
简而言之就是：参数更少、计算量更小、效果更好。

<div align = center> <img src="pictures/EffecienFPN_3.png "/></div>

现在，我们对于 FPN_base 的结构的介绍就告一段落。
下面开始介绍创新点在 **decoder 的设计** 上的结构

### Branch 2 Decoder_focus 结构

这部分的网络的特点是：encoder采用CNN、ResNet等经典的backbone，（在encoder中引入Residual_like结构，像UNet那样），主要贡献和研究点在decoder的设计上（有可学习的decoder，也有融合各种decoder特征的decoder）

这部分的网络结构有：
> + DUpsample
> + bilinear additive upsampling（+conv）
> + NRD

#### DUsample

（对特征的融合的操作的思想与 EffecientFPN 中的 decoder 如出一辙）

作者提出了一种参数可学习的 **上采样** 操作。该上采样的基本想法是将 groundtruth 的图变到 CNN　backbone　的最后一层特征图大小。

作者提出的 decoder 的结构为：

<div align = center> <img src="pictures/DUpsampling_3.png "/></div>

**解释**：

*一个重要的前提假设：segmatic segmentation label Y 中像素的分布并不是独立同分布的，每一个像素的值与周围像素的值是有一定关系的，所以我们可以将原始尺寸的 Y 压缩为小尺寸的向量而不导致信息的过多丢失。* 设$ F $ 是CNN　backbone　的最后一层特征图，$ F \in R^{\tilde{H} \times \tilde{W} \times \tilde{C}} $, 我们将 $ Y\in R^{H \times W \times C} $ 压缩到 $ \tilde{Y} \in R^{\tilde{H} \times \tilde{W} \times \tilde{C}} $ 的大小，然后尽量减少 $ F $ 和 $ \tilde{Y} $ 的差异，使用如下loss：

$$ L(F, Y) = ||F - \tilde{Y}||^2 $$

从 $ Y $ 压缩到 $ \tilde{Y} $ 的具体过程为：
① 设 $ r $ 为$ Y $ 到 $ \tilde{Y} $ 的比率，在论文中为 16 或者 32 （但是我认为如果不止用一次 DUpsampling 而是每次以 $ r = 2 $ 进行多次上采样，效果会更好）
② 我们将 $ Y $ 分为 $ r \times r \times C $ 的多个子窗口，对于每一个子窗口 $ S \in \{0, 1 \}^{r \times r \times C} $, 我们将 $ S $ reshape 为一个向量 $ v \in \{0, 1 \}^{N} $ ,其中 $ N = r \times r \times C $, 最终，我们将 $ v $ 压缩为一维向量 $ x \in R^{\tilde{C}} $, 然后将所有的 $ x $ 向量组合起来形成 $ \tilde{Y} $.
对于 $ v $ 到 $ x $ 的变换，可以直接使用一个线性变换：
$$ x = Pv; \tilde{v} = Wx $$
其中 $ P \in R^{\tilde{C} \times N} $, 用于将 $ v $ 变换到 $ x $，$ W \in R^{N \times \tilde{C}} $ 用于从 $ \tilde{Y} $ 到 $ Y $ 的转换（用于得到最终的分类图）（损失是在 CNN 最后一层进行的，但是还是得得到最终的分类图）

整个训练的流程为：
① **训练得到 encoder 网络的参数和 $ P $ 的参数**：用可学习的 $ P $ 将 $ Y $ 映射到 $ \tilde{H} \times \tilde{W} \times \tilde{C} $ 大小，使用 $ L(F, Y) = ||F - \tilde{Y}||^2 $ 作为 loss 进行训练，得到 encoder 网络的参数和 $ P $ 的参数。
② **得到 $ W $ 的参数**：$ W^* = \mathop{\arg\min}\limits_{W} = \sum ||v - \tilde{v}||^2 = \mathop{\arg\min}\limits_{W} = \sum ||v - WPv||^2$


*（挖坑：可不可以不 reshape，直接在 $ r \times r \times C$ 尺寸的张量上进行操作，我的直觉告诉我这样做的话效果会更好）*
*（挖坑：如果不止用一次 DUpsampling 而是每次以 $ r = 2 $ 进行多次上采样，效果可能会更好）*

值得注意的是，作者在网络中用的**特征融合**的操作和 EffecienFPN 中特征融合的操作有异曲同工之妙：将所有尺寸的特征图下采样到同一尺寸然后 concat，在我看来，这里的操作将各尺寸的特征图平等化（*虽然作者的出发点不是这个，作者认为这样融合特征的好处是：得益于 DUpsampling 的有效性，我们可以将各尺寸的特征都下采样到最小的特征尺寸大小，这样可以减少参数量和计算开销*，但是我认为**平等化各尺寸特征**更是其使得其效果更好的作用。

作者融合特征的操作如下图所示：

<div align = center> <img src="pictures/DUpsampling_1.png "/></div>

作者提出的 DUpsampling 的效果是优于 bilinear 的，如图：

 <div align = center> <img src="pictures/DUpsampling_2.png "/></div>

 

*（挖坑：上面的特征融合操作将各尺寸的特征平等化，但是我们是否可以设置一个可学习的权重参数呢？）*

>补充：这篇文章里作者提出的前提假设很有意思，可供我们以后的科研上参考：segmatic segmentation label Y 中像素的分布并不是独立同分布的，每一个像素的值与周围像素的值是有一定关系的，所以我们可以将原始尺寸的 Y 压缩为小尺寸的向量而不导致信息的过多丢失。不仅仅是 segmentation, 大部分的稠密预测任务都具有这样的特点

#### bilinear additive upsampling（+conv）

bilinear additive upsampling（+conv）的结构如下图所示：

<div align = center> <img src="pictures/bau_1.png "/></div>

上采样操作步骤：
① 对尺寸为 [H,W] 的 D 维通道的特征图进行逐通道维的双线性插值
② 将在上一步中得到的尺寸为 [2H,2W] 的 D 维通道的特征图进行分组：每四个通道为一组，对每组的四个通道上的特征图进行平均操作，最终得到尺寸为 [2H,2W] 的 $\frac{D}{4}$ 维通道的特征图，将其命名为 "identity"
③对 identity 进行卷积操作，这一步中存在可学习的参数
④对 ③ 中卷积后的特征图与 identity 进行 residual 连接（相加或者 concat），得到最终的上采样的 [2H,2W,D/4]（相加）或[2H,2W,D/2]（concat）的特征图。

bilinear additive upsampling（+conv）的效果如下图所示：

<div align = center> <img src="pictures/bau_2.png "/></div>

该结构在多数任务上都取得了较好的效果（接近甚至超过 SOTA）。

#### NRD

loading...

### Branch 3 U_Net_Pro

这部分的网络结构就是对 U_Net 的改进，主要表现为对 U_Net 中的 decoder 的改进（个人认为这个方向已经没有继续做下去的必要了，因为操作空间有限）

#### RefineNet

整体的网络结构为：

 <div align = center> <img src="pictures/RefineNet_1.png "/></div>

 

RefineNet 结构的具体细节为：

 <div align = center> <img src="pictures/RefineNet_2.png "/></div>
 

### 其他网络结构

#### Stacked Hourglass Networks

网络的整体结构如下图所示：
网络由多个串联的沙漏状的 encoder_decoder 结构组成。

 <div align = center> <img src="pictures/hourglass_1.png "/></div>
 

我在看了代码之后，将整个网络的结构更细节地梳理了出来：
整个网络结构可分为三个层次：

①**第一层次**：最基本的 Residual 模块：

 
<div align = center> <img src="pictures/hourglass_2.png "/></div>
 

解释：该模块由 **Relu**、**Batch Normonization**、**$ 1 \times 1$ convolution**、**$ 3 \times 3$ convolution**、**残差链接** 组成

②**第二层次**：hourglass 模块

 <div align = center> <img src="pictures/hourglass_3.png "/></div>
 

解释：该模块由 **Residual 模块**、**下采样**、**上采样** 组成

③**第三层次**：整个网络

 <div align = center> <img src="pictures/hourglass_4.png "/></div>
 

解释：该模块由 **hourglass 模块**、**Residual 模块**、**下采样**、**$ 1 \times 1$ convolution**、**$ 7 \times 7$ convolution**、**残差连接** 组成。在每一个 **hourglass 模块** 后都进行 loss 的计算，以优化对网络参数的学习。

# chapter 2 重要部件介绍

## Section 1 转置卷积

>本小结转载于 [博客](https://blog.csdn.net/qq_39478403/article/details/121181904)

曾经，转置卷积也被称为 **反卷积** (Deconvolution)。与传统的上采样方法相比，转置卷积的上采样方式并非预设的插值方法，而是同标准卷积一样，**具有可学习的参数**，可通过网络学习来获取最优的上采样方式。

### 转置卷积与标准卷积的区别

标准卷积操作实际上就是建立了一个 **多对一的映射关系**。

对转置卷积而言，我们实际上是想建立一个逆向操作，即 一对多的映射关系。对于上例，我们想要建立的其实是输出矩阵中的 1 个值与输入矩阵中的 9 个值的关系，如下图所示：

 <div align = center> <img src="pictures/zzjj_1.png "/></div>
 

当然，从信息论的角度上看，常规卷积操作是不可逆的，所以转置卷积并不是通过输出矩阵和卷积核计算原始输入矩阵，而是计算得到保持了相对位置关系的矩阵。

### 转置卷积的推导

定义一个 **4×4 输入矩阵 input**：

 <div align = center> <img src="pictures/zzjj_2.png "/></div>
 

再定义一个 **3×3 标准卷积核 kernel**：

 <div align = center> <img src="pictures/zzjj_3.png "/></div>
 

设 步长 stride=1、填充 padding=0，则按 "valid" 卷积模式，可得 **2×2 输出矩阵 output**：

 <div align = center> <img src="pictures/zzjj_4.png "/></div>
 

这里，换一个表达方式，将输入矩阵 input 和输出矩阵 output 展开成 **16×1 列向量 X 和 4×1 列向量 Y**，可分别表示为：

 <div align = center> <img src="pictures/zzjj_5.png "/></div>
 

接着，再用矩阵运算来描述标准卷积运算，设有 **新卷积核矩阵 C**：

 <div align = center> <img src="pictures/zzjj_6.png "/></div>
 

经推导 (卷积运算关系)，可得 **4×16 稀疏矩阵 C**：

<div align = center> <img src="pictures/zzjj_7.png "/></div>

以下，用下图展示矩阵运算过程：

 <div align = center> <img src="pictures/zzjj_8.png "/></div>
 

而转置卷积其实就是要对这个过程进行逆运算，即 **通过 C 和 Y 得到 X**：

 <div align = center> <img src="pictures/zzjj_9.png "/></div>
 

此时，$ C^T $ 即为新的 **16×4 稀疏矩阵**。以下通过下图展示转置后的卷积矩阵运算。此处，**用于转置卷积的权重矩阵**  $ C^T $ 不一定来自于原卷积矩阵 $ C $ (通常不会如此恰巧)，但其形状和原卷积矩阵 $ C $ 的转置相同。

 <div align = center> <img src="pictures/zzjj_10.png "/></div>
 

最后，将 16×1 的输出结果重新排序，即可通过 2×2 输入矩阵得到 4×4 输出矩阵。

### 转置卷积的输出

#### stride = 1 

同样，使用上文中的 **3×3 卷积核矩阵 C**：

<div align = center> <img src="pictures/zzjj_11.png "/></div>
 

**输出矩阵 output** 仍为：

<div align = center> <img src="pictures/zzjj_12.png "/></div>
 

将输出矩阵展开为 **列向量 Y**：

 <div align = center> <img src="pictures/zzjj_13.png "/></div>
 

带入到上文中的转置卷积计算公式，则转置卷积的计算结果为：

 <div align = center> <img src="pictures/zzjj_14.png "/></div>
 

这其实等价于 **先填充 padding=2 的输入矩阵 input**：

 <div align = center> <img src="pictures/zzjj_15.png "/></div>
 

然后，**转置标准卷积核 kernel**：

 <div align = center> <img src="pictures/zzjj_16.png "/></div>
 

最后，在 **填零的输入矩阵 input** 上使用 **经转置的标准卷积核 kernel** 执行 **标准卷积运算**，如下图所示：

 <div align = center> <img src="pictures/zzjj_17.png "/></div>
 

更一般地，对于卷积核尺寸 kernel size = $ k $，步长 stride = $ s $ = 1，填充 padding = $ p $ = 0 的转置卷积，其 **等价的标准卷积** 在原尺寸为 $ p' $ 的输入矩阵上进行运算，输出特征图的尺寸 $ o' $ 为：

$$ o' = (i' _1) + k $$

同时，等价的标准卷积的的输入矩阵 input 在卷积运算前，需先进行 padding' = $k _ 1 $ 的填充，得到尺寸 $ i'' = i' + 2(k_1) $。

因此，实际上原计算公式为 (等价的标准卷积的步长 $ s' = 1 $)：

$$ o'= \frac{i'' _ k + 2p }{s'} + 1 = i' + 2(k _ 1) _k + 1 = (i' _ 1) + k $$


#### stride > 1

在实际中，我们大多数时候会使用 stride>1 的转置卷积，从而获得较大的上采样倍率。

以下，令输入尺寸为 **5×5**，标准卷积核同上  kernel size = $k$ = **3**，步长  stride = $ s $ = 2，填充  padding = $ p $ = 0，标准卷积运算后，输出矩阵尺寸为 **2×2**：

 <div align = center> <img src="pictures/zzjj_18.png "/></div>
 

此处，转换后的稀疏矩阵尺寸变为 25×4，由于矩阵太大这里不展开进行罗列。最终转置卷积的结果为：

 <div align = center> <img src="pictures/zzjj_19.png "/></div>
 

此时，等价于 **输入矩阵同时添加了 空洞 和 填充**，再由仅转置的标准卷积核进行运算，过程如图 7 所示：

 <div align = center> <img src="pictures/zzjj_20.png "/></div>
 

更一般地，对于卷积核尺寸 kernel size = $k$，步长 stride = $s$ > 1，填充 padding = $p$ = 0 的转置卷积，其 等价的标准卷积 在原尺寸为 $i'$ 的输入矩阵上进行运算，输出特征图的尺寸 $o'$ 为： 

$$ o' = s(i' _ 1) + k $$

同时，等价的标准卷积的输入矩阵 input 在卷积运算前，需要先进行 padding' = $ k _ 1 $ 的填充；然后，相邻元素的空洞数为 $ s _ 1 $，共有 $ i' _ 1 $ 组空洞需要插入；从而，实际尺寸为

$$ i'' = i' + 2(k _ 1) + (i' _ 1) \times (s _ 1) = s \times (i' _ 1) + 2k _1 $$

因此，实际上原计算公式为（等价的标准卷积的步长 $ s' = 1$）:

$$ o'  = \frac{i'' - k + 2p}{s'} + 1 =s(i' - 1) + 2k -1 - k + 1 = s(i' _ 1) + k $$

可见，**通过控制步长 stride = s 的大小可以控制上采样的倍率**，而该参数类比于膨胀/空洞卷积的 **膨胀率/空洞数**。

#### 小结

注意：矩阵中的实际权值不一定来自原始卷积矩阵。重要的是权重的排布是由卷积矩阵的转置得来的。转置卷积运算与普通卷积形成相同的连通性，但方向是反向的。

我们可以用转置卷积来上采样，而 **转置卷积的权值是可学习的**，所以无需一个预定义的插值方法。

**尽管它被称为转置卷积，但这并不意味着我们取某个已有的卷积矩阵并使用转置后的版本**。重点是，与标准卷积矩阵 (一对多关联而不是多对一关联) 相比，输入和输出之间的关联是以反向的方式处理的

因此，**转置卷积不是卷积，但可以用卷积来模拟转置卷积。通过在输入矩阵的值间插入零值 (以及周围填零) 上采样输入矩阵，然后进行常规卷积 就会产生 与转置卷积相同的效果**。你可能会发现一些文章用这种方式解释了转置卷积。但是，由于需要在常规卷积前对输入进行上采样，所以效率较低。 

注意：转置卷积会导致生成图像中出现 **网格/棋盘效应 (checkerboard artifacts)**，因此后续也存在许多针对该问题的改进工作。


## Section 2 对4种 decoder 的考察

> 本小节来源于 **the devil is in the decoder** 这篇文章

本小节考察的 decoder 结构有：
> Transposed Convolution （转置卷积）
> Decomposed Transposed Convolution（分解转置卷积）
> Depth_To_Space
> bilinear

### Transposed Convolution （转置卷积）

由前面的章节我们已经提到过：转置卷积可以等效为 **填充之后的标准卷积**，下面我们再次给出转置卷积的结构，并进一步讨论 **转置卷积 output 中的每一个像素的信息来源于 input 中的哪些像素**。

首先，转置卷积的结构如下图所示：

 <div align = center> <img src="pictures/TC_1.png "/></div>
 

转置卷积 output 中的每一个像素中的信息对于 input 中像素的依赖关系为：

 <div align = center> <img src="pictures/TC_2.png "/></div>
 

由上图可以看出：转置卷积 output 中的每一个像素中的信息并不是均匀的，**而正是这种信息的不均匀，导致了 棋盘效应 的出现**

### Decomposed Transposed Convolution（分解转置卷积）

分解转置卷积与转置卷积思路相同，但是在处理上有区别：**分解转置卷积在 x、y 两个维度分开填充 0，这样可以使得需要学习的参数量更少**，分解转置卷积的结构和其 output 对于 input 的信息依赖关系如图所示：

 <div align = center> <img src="pictures/DTC_1.png "/></div>
 

由上图可以看出：虽然分解转置卷积与转置卷积相比，具有更少的参数量，但是分解转置卷积仍然会有 **棋盘效应**

### Depth-To-Space

 <div align = center> <img src="pictures/DTS_1.png "/></div>
 

该 decoder 将 4 个小尺寸的特征图拼接起来，形成一个大尺寸的特征图。

### bilinear

双线性插值有两个模式：

① align_corners = True

output 和 input 的角上的像素对其，如下图所示：

 <div align = center> <img src="pictures/acT_1.png "/></div>
 

示例：

```python
tensor([[[[1., 2.],
          [3., 4.]]]])
m = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
m(input)
```
输出为：
```python
tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
          [1.6667, 2.0000, 2.3333, 2.6667],
          [2.3333, 2.6667, 3.0000, 3.3333],
          [3.0000, 3.3333, 3.6667, 4.0000]]]])
```


② align_corners = False

utput 和 input 的角上的像素对其，如下图所示：

 <div align = center> <img src="pictures/acF_1.png "/></div>
 

示例：

```python
tensor([[[[1., 2.],
          [3., 4.]]]])
m = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=Flase)
m(input)
```
输出为：
```python
tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
          [1.5000, 1.7500, 2.2500, 2.5000],
          [2.5000, 2.7500, 3.2500, 3.5000],
          [3.0000, 3.2500, 3.7500, 4.0000]]]])
```

### 总结

**the devil is in the decoder** 这篇文章的作者在介绍了以上几种 decoder 的结构之后，比较了他们的效果，最后得出的结论为：

> 1. 在做比较的过程中，用了 with residual_like connections 和 without residual_like connections 的结构，发现 with residual_like connections 的结构效果更好
> 2. 整体来说，transposed convolutions, depth_to_space, and bi_linear additive upsampling 这三个 decoder 的效果最好


## Section 3 对空洞卷积的改进

>本小节转载于 [博客](https://zhuanlan.zhihu.com/p/50369448)

空洞卷积是存在理论问题的，论文中称为gridding，其实就是网格效应/棋盘问题。因为空洞卷积得到的某一层的结果中，邻近的像素是从相互独立的子集中卷积得到的，相互之间缺少依赖。

+ **局部信息丢失**：由于空洞卷积的计算方式类似于棋盘格式，某一层得到的卷积结果，来自上一层的独立的集合，没有相互依赖，因此该层的卷积结果之间没有相关性，即局部信息丢失。
+ 远距离获取的信息没有相关性：由于空洞卷积稀疏的采样输入信号，使得远距离卷积得到的信息之间没有相关性，影响分类结果。

### 解决方案

+ Panqu Wang,Pengfei Chen, et al.**Understanding Convolution for Semantic Segmentation**.//WACV 2018

 <div align = center> <img src="pictures/kdjj_1.png "/></div>
 

通过上图a解释了空洞卷积存在的问题，从左到右属于top_bottom关系，三层卷积均为r=2的dilatedConv,可以看出最上层的红色像素的感受野为13且参与实际计算的只有75%，很容易看出其存在的问题。

使用HDC的方案解决该问题，不同于采用相同的空洞率的deeplab方案，**该方案将一定数量的layer形成一个组，然后每个组使用连续增加的空洞率，其他组重复**。如deeplab使用rate=2,而HDC采用r=1,r=2,r=3三个空洞率组合，这两种方案感受野都是13。但HDC方案可以从更广阔的像素范围获取信息，避免了grid问题。同时该方案也可以通过修改rate任意调整感受野。

+ Fisher Yu, et al. **Dilated Residual Networks**. //CVPR 2017

 <div align = center> <img src="pictures/kdjj_2.png "/></div>
 

如果特征map有比空洞率更高频的内容，则grid问题更明显。

提出了三种方法：

**Removing max pooling**：由于maxpool会引入更高频的激活，这样的激活会随着卷积层往后传播，使得grid问题更明显。

 <div align = center> <img src="pictures/kdjj_3.png "/></div>
 

**Adding layers**：在网络最后增加更小空洞率的残参block, 有点类似于HDC。

**Removing residual connections**：去掉残参连接，防止之前层的高频信号往后传播。

+ Zhengyang Wang,et al.**Smoothed Dilated Convolutions for Improved Dense Prediction**.//KDD 2018.

 <div align = center> <img src="pictures/kdjj_4.png "/></div>
 

空洞卷积的分解观点，在原始特征图上周期性采样形成4组分辨率降低的特征图，然后使用原始的空洞卷积参数(去掉了空洞0)分别进行卷积，之后将卷积的结果进行上采样组合。**从该分解观点可以看出，卷积前后的4个组之间没有相互依赖，使得收集到不一致的局部信息**

 <div align = center> <img src="pictures/kdjj_5.png "/></div>
 

从上面分解的观点出发：

(1) 在最后生成的4组卷积结果之后，**经过一层组交错层**，类似于ShuffleNet，使得每组结果能进行相互交错，相互依赖，以此解决局部信息不一致的问题。

 <div align = center> <img src="pictures/kdjj_6.png "/></div>

<div align = center> <img src="pictures/Fkdjj_7.png "/></div>
 

(2) 第二种方法为在空洞卷积之前进行局部信息依赖，即增加一层卷积操作，卷积利用了分离卷积，并且所有通道共享参数。

 <div align = center> <img src="pictures/kdjj_8.png "/></div>

<div align = center> <img src="pictures/kdjj_9.png "/></div>
 

+ Liang_Chieh Chen,et al.**Rethinking Atrous Convolution for Semantic Image Segmentation**//2017

deeplabv3在v2基础上进一步探索空洞卷积，分别研究了级联ASPP与并联ASPP两种结构。

 <div align = center> <img src="pictures/kdjj_10.png "/></div>
 

deeplabv3不同于deeplabv2，在resnet101基础上 **级联** 了更深的网络，随着深度的增加，使用了不同的空洞率的卷积，这些卷积保证分辨率不降低的情况下，感受野可以任意控制，一般让空洞率成倍增加。同时使用了Multigrid策略，在同一个blocks的不同层使用分层的空洞率，如2,4,8，而不是都使用2，这样使得感受野相比原来的有所增加。**但这样同样会存在grid问题**

 <div align = center> <img src="pictures/kdjj_11.png "/></div>
 

ASPP存在的问题，当使用的空洞率增大时，有效的滤波参数数量逐渐减小。极端的，当r等于特征图大小时，该卷积没有捕获整幅图像的上下文信息，而是退化为1*1卷积

解决方案：**增加图像级特征，使用全局池化获取图像全局信息**，而其他部分的卷积为了捕获多尺度信息，这里的卷积不同于deeplabv2，加了batch normalization。

+ Sachin Mehta,et al. ESPNet: **Efficient Spatial Pyramid of DilatedConvolutions for Semantic Segmentation**. //ECCV 2018

ESPNet利用分解卷积的思想，先用1*1卷积将通道数降低减少计算量，后面再加上基于空洞卷积的金字塔模型，捕获多尺度信息

 <div align = center> <img src="pictures/kdjj_12.png "/></div>
 

之前的方法都是通过引入新的计算量，学习新的参数来解决grid问题。**而这里直接使用了特征分层的思想直接将不同rate的空洞卷积的输出分层sum，其实就是将不同的感受野相加，弥补了空洞带来的网格效应**。从结果上看效果不错

 <div align = center> <img src="pictures/kdjj_13.png "/></div>
 

训练技巧：

所有卷积后都使用BN和PReLU，**很多实时分割小网络都使用了PReLU**；

使用Adam训练，很多小网络使用这个；

+ Tianyi Wu,et al.**Tree_structured Kronecker Convolutional Networks for Semantic Segmentation**.//ICME 2019

 <div align = center> <img src="pictures/kdjj_14.png "/></div>
 

使用Kronecker convolution来解决空洞卷积局部信息丢失问题，以r1=4、r2=3为例，KConv将每个标准卷积的元素都乘以一个相同的矩阵，该矩阵由0,1组成，这样参数量是不增加的。该矩阵为：

 <div align = center> <img src="pictures/kdjj_15.png "/></div>
 

这样每个元素乘以矩阵后变为上面右图所示的图。因此，可以看出r1控制空洞的数量，也即扩大了感受野，而r2控制的是每个空洞卷积忽视的局部信息。当r2=1时，其实就是空洞卷积，当r2=r1=1时就是标准卷积。

**总体效果mIOU提升了1%左右。**

除此之外，提出了一个TFA模块，利用树形分层结构进行多尺度与上下文信息整合。**结构简单，但十分有效，精度提升4_5%。**

+ Hyojin Park,et al.**Concentrated_Comprehensive Convolutionsfor lightweight semantic segmentation**.//2018

 <div align = center> <img src="pictures/kdjj_16.png "/></div>
 

针对实时语义分割提出的网络结构，深度分离卷积与空洞卷积的组合，在ESPNet上做的实验。并且说明简单的组合会带来精度的降低，由于局部信息的丢失。为此，在深度分离空洞卷积之前，使用了两级一维分离卷积捕获局部信息。

网络结构上与ESPNet保持一致，其中，并行分支结果直接Cat，不需要后处理，每个分支不需要bn+relu。消融实验表明，在一维卷积之间加入BN+PReLU，精度会增加1.4%。

## Section 4 Fast Normalized Fusion

> Fast Normalized Fusion 提出于 **BiFPN** 这篇文章

提出的 Fast Normalized Fusion 用以替代 Softmax_based fusion

Fast Normalized Fusion 效果为：

Fast normalized fusion的速度为 Softmax_based fusion 的1.3倍左右，但是效果几乎一样。如下图所示：

 <div align = center> <img src="pictures/FNF_1.png "/></div>
 

具体地，Fast Normalized Fusion 的结构为：

 <div align = center> <img src="pictures/FNF_2.png "/></div>
 

while，Softmax_based fusion 的结构为：

 <div align = center> <img src="pictures/FNF_3.png "/></div>
 

# chapter 3 自己的思考
在看了上述的 encoder-decoder 结构之后，我对于之后的研究有以下思考：

我觉得我比较看好的方向，或者是我之后可能会去研究的方向：（按优先级排序）
> ① 对带参数的 decoder 的研究，像 Dupsampling、NRD 这些文章中的 decoder 那样，考虑的点会有 output 中像素对 input 中像素的依赖关系：尽量使得这样的依赖关系是均匀的（消除棋盘效应），像 **the devil is in the decoder** 中作者那样去考察和思考 decoder 的结构。
> ② 使用 Dupsampling、NRD 这些文章中的 decoder 的单元结构来设计一个 decoder 网络（因为我觉得 这两篇文章中的 r = 32 太粗糙了）
> ③ 将 decoder_focus 这个小结中的 decoder 与 FPN 相结合，构造出一个网络（思路是：encoder 用最好的，decoder 也用最好的，缝合怪了属于是）
> ④ 去解决一下我在文章中写在 *（挖坑：）* 里的问题和内容

另外，我还有一些我觉得比较有意思的想法，但是这篇文章写在开源网站上，所以我就不在这篇文章里面提及了，我在 biweekly report 中写。

