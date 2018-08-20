package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link ConvolutionLayer} with activation followed by {@link BatchNormalization}
 *
 * @author Christian Skärby
 */
public class Conv2DBatchNormAfter implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Conv2DBatchNormAfter.class);

    private int nrofKernels = 64;
    private int kernelSize_h = 4;
    private int kernelSize_w = 4;
    private int stride_w = 1;
    private int stride_h = 1;
    private ConvolutionMode convolutionMode = ConvolutionMode.Truncate;

    private IActivation activation = new ActivationELU();
    private boolean strideChanged = false;

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        actStr = actStr.isEmpty() ? actStr : "_" + actStr;
        String kernelSize = kernelSize_h == kernelSize_w ? "" + kernelSize_h : kernelSize_h + "_" + kernelSize_w;
        final String strideStr = strideChanged ? stride_h == stride_w ? "_" + stride_h : "_" + stride_h + "_" + stride_w : "";
        return "C_" + nrofKernels + "_" + kernelSize + strideStr + actStr + "_BN";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("cnn Layer: " + info);
        BlockInfo nextInfo = builder
                .layer(info, new ConvolutionLayer.Builder(kernelSize_h, kernelSize_w)
                        .stride(stride_h, stride_w)
                        .activation(activation)
                        .nOut(nrofKernels)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(convolutionMode)
                        .build());
        log.info("cnn BN Layer: " + nextInfo);
        nextInfo = builder
                .layer(nextInfo, new BatchNormalization.Builder()
                        .eps(1e-3)
                        .build());
        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(nrofKernels)
                .build();
    }

    /**
     * Sets the number of kernels (==nrof output maps)
     *
     * @param nrofKernels the number of kernels
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setNrofKernels(int nrofKernels) {
        this.nrofKernels = nrofKernels;
        return this;
    }

    /**
     * Convenience method which sets both kernel height and width to the given value
     *
     * @param kernelSize the kernelSize
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setKernelSize(int kernelSize) {
        this.kernelSize_h = kernelSize;
        this.kernelSize_w = kernelSize;
        return this;
    }

    /**
     * Sets kernel height
     *
     * @param kernelSize_h the kernel height
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setKernelSize_h(int kernelSize_h) {
        this.kernelSize_h = kernelSize_h;
        return this;
    }

    /**
     * Sets kernel width
     *
     * @param kernelSize_w the kernel width
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setKernelSize_w(int kernelSize_w) {
        this.kernelSize_w = kernelSize_w;
        return this;
    }

    /**
     * Convenience method which sets both height and width stride to the given value
     *
     * @param stride the stride
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setStride(int stride) {
        setStride_w(stride);
        setStride_h(stride);
        return this;
    }

    /**
     * Sets the width stride
     *
     * @param stride_w the width stride
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setStride_w(int stride_w) {
        strideChanged = true;
        this.stride_w = stride_w;
        return this;
    }

    /**
     * Sets the height stride
     *
     * @param stride_h the height stride
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setStride_h(int stride_h) {
        strideChanged = true;
        this.stride_h = stride_h;
        return this;
    }

    /**
     * Sets the activation function to use
     *
     * @param activation the activation function to use
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setActivation(IActivation activation) {
        this.activation = activation;
        return this;
    }

    /**
     * Sets the {@link ConvolutionMode} to use
     *
     * @param convolutionMode the mode to use
     * @return the {@link Conv2DBatchNormAfter}
     */
    public Conv2DBatchNormAfter setConvolutionMode(ConvolutionMode convolutionMode) {
        this.convolutionMode = convolutionMode;
        return this;
    }
}
