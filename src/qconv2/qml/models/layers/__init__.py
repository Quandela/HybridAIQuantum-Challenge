from .classical.add import AddBuilder
from .classical.concat import ConcatBuilder
from .classical.mul import MulBuilder
from .classical.repeat import RepeatBuilder
from .classical.reshape import ReshapeBuilder
from .classical.sum import SumBuilder
from .classical.flatten import FlattenBuilder
from .classical.batch_norm_1d import BatchNorm1dBuilder
from .classical.batch_norm_2d import BatchNorm2dBuilder
from .classical.average_pool_2d import AvgPool2dBuilder
from .classical.conv_1d import Conv1dBuilder
from .classical.conv_2d import Conv2dBuilder
from .classical.deconv_1d import Deconv1dBuilder
from .classical.deconv_2d import Deconv2dBuilder
from .classical.pixelshuffle import PixelShuffleBuilder
from .classical.group_norm import GroupNormBuilder
from .classical.instance_norm import InstanceNormBuilder
from .classical.leaky_relu import LeakyReLUBuilder
from .classical.linear import LinearBuilder
from .classical.max_pool_2d import MaxPool2dBuilder
from .classical.subpixelconv_2d import SubpixelConv2dBuilder
from .classical.prelu import PReLUBuilder
from .classical.relu import ReLUBuilder
from .classical.mish import MishBuilder
from .classical.resblock import ResblockBuilder
from .classical.softmax import SoftmaxBuilder
from .classical.sigmoid import SigmoidBuilder
from .classical.fourier_features_2d import FourierFeatures2dBuilder
from .classical.fourier_features_1d import FourierFeatures1dBuilder
from .classical.pca import PCABuilder
from .classical.identity import IdentityBuilder
from .quantum.boson_sampling import BosonSamplingBuilder
from .quantum.photonic_circuit import PhotonicCircuitBuilder
from .quantum.slos_circuit import SLOSCircuitBuilder
from .quantum.qconv_2d import QuantumConv2dBuilder

# Standard layers
resblock = ResblockBuilder.elem
add = AddBuilder.elem
mul = MulBuilder.elem
sum = SumBuilder.elem
concat = ConcatBuilder.elem
reshape = ReshapeBuilder.elem
repeat = RepeatBuilder.elem
flatten = FlattenBuilder.elem
pixelshuffle = PixelShuffleBuilder.elem
subpixelconv2d = SubpixelConv2dBuilder.elem
fourier_features_2d = FourierFeatures2dBuilder.elem
fourier_features_1d = FourierFeatures1dBuilder.elem
linear = LinearBuilder.elem
conv_1d = Conv1dBuilder.elem
conv_2d = Conv2dBuilder.elem
deconv_1d = Deconv1dBuilder.elem
deconv_2d = Deconv2dBuilder.elem
relu = ReLUBuilder.elem
mish = MishBuilder.elem
lrelu = LeakyReLUBuilder.elem
prelu = PReLUBuilder.single
softmax = SoftmaxBuilder.elem
sigmoid = SigmoidBuilder.elem
batchnorm_1d = BatchNorm1dBuilder.elem
batchnorm_2d = BatchNorm2dBuilder.elem
groupnorm = GroupNormBuilder.elem
instancenorm = InstanceNormBuilder.elem
maxpool_2d = MaxPool2dBuilder.elem
avgpool_2d = AvgPool2dBuilder.elem
identity = IdentityBuilder.elem

# Quantum layers
boson_sampling = BosonSamplingBuilder.elem
photonic_circuit = PhotonicCircuitBuilder.elem
slos_circuit = SLOSCircuitBuilder.elem
qconv_2d = QuantumConv2dBuilder.elem

# Pretrained layers
pca = PCABuilder.elem


### Tools
def duplicate(sequence: list, duplication: int) -> dict:
    assert int(duplication)

    duplications = []

    if not isinstance(sequence, list):
        sequence = [sequence]

    for _ in range(duplication):
        for mod in sequence:
            duplications.append(mod)

    return duplications
