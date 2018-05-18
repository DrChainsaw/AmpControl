package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Optional;

/**
 * Adds a {@link Convolution1DLayer}. Can also set an {@link InputPreProcessor}.
 *
 * @author Christian Skärby
 */
public class Conv1D implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Conv1D.class);
    
    private int nrofKernels = 64;
    private int kernelSize = 4;
    private int stride = 1;
    private Optional<InputPreProcessor> inputPreProcessorOptional = Optional.empty();
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.NO_WORKSPACE;

    private IActivation activation = new ActivationELU();
    private boolean strideChanged = false;

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        actStr = actStr.isEmpty() ? actStr : "_" + actStr;
        final String strideStr = strideChanged ? "_" + stride : "";
        return "C1_" + nrofKernels + "_" + kernelSize + strideStr + actStr;
    }

    @Override
    public BlockInfo addLayers(NeuralNetConfiguration.ListBuilder listBuilder, BlockInfo info) {
        final BlockInfo thisLayerInfo = LayerBlockConfig.super.addLayers(listBuilder, info);
        inputPreProcessorOptional.ifPresent(preproc -> listBuilder.inputPreProcessor(thisLayerInfo.getPrevLayerInd(), preproc));
        return thisLayerInfo;
    }

    @Override
    public BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, BlockInfo info) {
        final BlockInfo thisLayerInfo = LayerBlockConfig.super.addLayers(graphBuilder, info);
        Arrays.stream(thisLayerInfo.getInputsNames()).forEach(layerName -> {
            inputPreProcessorOptional.ifPresent(preproc -> graphBuilder.inputPreProcessor(layerName, preproc));
        });
        return thisLayerInfo;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("cnn Layer: " + info);
        BlockInfo nextLayer = builder
                .layer(info, new Convolution1DLayer.Builder(kernelSize)
                        .stride(stride)
                        .activation(activation)
                        .nOut(nrofKernels)
                        .cudnnAlgoMode(cudnnAlgoMode)
                        .build());
        return new SimpleBlockInfo.Builder(nextLayer)
                .setPrevNrofOutputs(nrofKernels)
                .build();
    }

    /**
     * Sets the number of kernels (==nrof output maps)
     * @param nrofKernels
     * @return the {@link Conv1D}
     */
    public Conv1D setNrofKernels(int nrofKernels) {
        this.nrofKernels = nrofKernels;
        return this;
    }

    /**
     * Sets the size (length) of each filter kernel
     * @param kernelSize
     * @return the {@link Conv1D}
     */
    public Conv1D setKernelSize(int kernelSize) {
        this.kernelSize = kernelSize;
        return this;
    }

    /**
     * Sets the filter stride
     * @param stride
     * @return the {@link Conv1D}
     */
    public Conv1D setStride(int stride) {
        this.strideChanged = true;
        this.stride = stride;
        return this;
    }

    /**
     * Sets the activation function to use
     * @param activation
     * @return the {@link Conv1D}
     */
    public Conv1D setActivation(IActivation activation) {
        this.activation = activation;
        return this;
    }

    /**
     * Sets the input preprocessor
     * @param inputPreProcessor
     * @return  the {@link Conv1D}
     */
    public Conv1D setInputPreproc(InputPreProcessor inputPreProcessor) {
        inputPreProcessorOptional = Optional.of(inputPreProcessor);
        return this;
    }
}
