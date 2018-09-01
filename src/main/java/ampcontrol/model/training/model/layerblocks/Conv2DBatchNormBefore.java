package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link BatchNormalization} layer with activation followed by {@link ConvolutionLayer} without activation.
 *
 * @author Christian Skärby
 */
public class Conv2DBatchNormBefore implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Conv2DBatchNormBefore.class);

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
        final String kernelSize = kernelSize_h == kernelSize_w ? "" + kernelSize_h : kernelSize_h + "_" + kernelSize_w;
        final String strideStr = strideChanged ? stride_h == stride_w ? "_" + stride_h : "_" + stride_h + "_" + stride_w : "";
        return "BN" + actStr + "_C_" + nrofKernels + "_" + kernelSize + strideStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("cnn BN Layer: " + info);
        BlockInfo nextInfo = builder
                .layer(info, new BatchNormalization.Builder()
                        .eps(1e-3)
                        .activation(activation)
                        .build());

        log.info("cnn Layer: " + nextInfo);
        nextInfo = builder
                .layer(nextInfo, new ConvolutionLayer.Builder(kernelSize_h, kernelSize_w)
                        .nOut(nrofKernels)
                        .activation(new ActivationIdentity())
                        .stride(stride_h, stride_w)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(convolutionMode)
                        .build());
        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(nrofKernels)
                .build();
    }

    /**
     * Sets the number of kernels (==nrof output maps)
     *
     * @param nrofKernels
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setNrofKernels(int nrofKernels) {
        this.nrofKernels = nrofKernels;
        return this;
    }

    /**
     * Convenience method which sets both kernel height and width to the given value
     *
     * @param kernelSize
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setKernelSize(int kernelSize) {
        this.kernelSize_h = kernelSize;
        this.kernelSize_w = kernelSize;
        return this;
    }

    /**
     * Sets kernel height
     *
     * @param kernelSize_h
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setKernelSize_h(int kernelSize_h) {
        this.kernelSize_h = kernelSize_h;
        return this;
    }

    /**
     * Sets kernel width
     *
     * @param kernelSize_w
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setKernelSize_w(int kernelSize_w) {
        this.kernelSize_w = kernelSize_w;
        return this;
    }

    /**
     * Convenience method which sets both height and width stride to the given value
     *
     * @param stride
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setStride(int stride) {
        setStride_w(stride);
        setStride_h(stride);
        return this;
    }

    /**
     * Sets the width stride
     *
     * @param stride_w
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setStride_w(int stride_w) {
        strideChanged = true;
        this.stride_w = stride_w;
        return this;
    }

    /**
     * Sets the height stride
     *
     * @param stride_h
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setStride_h(int stride_h) {
        strideChanged = true;
        this.stride_h = stride_h;
        return this;
    }

    /**
     * Sets the activation function to use
     *
     * @param activation
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setActivation(IActivation activation) {
        this.activation = activation;
        return this;
    }

    /**
     * Sets the {@link ConvolutionMode} to use
     *
     * @param convolutionMode the mode to use
     * @return the {@link Conv2DBatchNormBefore}
     */
    public Conv2DBatchNormBefore setConvolutionMode(ConvolutionMode convolutionMode) {
        this.convolutionMode = convolutionMode;
        return this;
    }
}
