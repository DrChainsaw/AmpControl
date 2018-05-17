package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link ConvolutionLayer}
 * 
 * @author Christian Skärby
 */
public class Conv2D implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Conv2D.class);

    private int nrofKernels = 64;
    private int kernelSize_h = 4;
    private int kernelSize_w = 4;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.NO_WORKSPACE;

    private IActivation activation = new ActivationELU();

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        actStr = actStr.isEmpty() ? actStr : "_" + actStr;
        String kernelSize = kernelSize_h == kernelSize_w ? "" + kernelSize_h : kernelSize_h + "_" + kernelSize_w;
        return "C_" + nrofKernels + "_" + kernelSize + actStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("cnn Layer: " + info);
        BlockInfo nextInfo = builder
                .layer(info, new ConvolutionLayer.Builder(kernelSize_h, kernelSize_w)
                        .activation(activation)
                        .nOut(nrofKernels)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .build());
        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(nrofKernels)
                .build();
    }

    /**
     * Sets the number of kernels (==nrof output maps)
     * @param nrofKernels
     * @return the {@link Conv2D}
     */
    public Conv2D setNrofKernels(int nrofKernels) {
        this.nrofKernels = nrofKernels;
        return this;
    }

    /**
     * Convenience method which sets both kernel height and width to the given value
     * @param kernelSize
     * @return the {@link Conv2D}
     */
    public Conv2D setKernelSize(int kernelSize) {
        this.kernelSize_h = kernelSize;
        this.kernelSize_w = kernelSize;
        return this;
    }

    /**
     * Sets kernel height
     * @param kernelSize_h
     * @return the {@link Conv2D}
     */
    public Conv2D setKernelSize_h(int kernelSize_h) {
        this.kernelSize_h = kernelSize_h;
        return this;
    }

    /**
     * Sets kernel width
     * @param kernelSize_w
     * @return the {@link Conv2D}
     */
    public Conv2D setKernelSize_w(int kernelSize_w) {
        this.kernelSize_w = kernelSize_w;
        return this;
    }

    /**
     * Sets the activation function
     * @param activation
     * @return the {@link Conv2D}
     */
    public Conv2D setActivation(IActivation activation) {
        this.activation = activation;
        return this;
    }
}
