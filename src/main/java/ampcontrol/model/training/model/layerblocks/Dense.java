package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds a {@link DenseLayer}.
 * 
 * @author Christian Skärby
 */
public class Dense implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(Dense.class);

    private int hiddenWidth = 256;
    private IActivation activation = new ActivationReLU();

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        return "dnn_" +  hiddenWidth + "_" + actStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("dnn Layer: " + info);
        BlockInfo nextInfo = builder
                .layer(info, new DenseLayer.Builder()
                        .activation(activation)
                        .nOut(hiddenWidth)
                        .build());
        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(hiddenWidth)
                .build();
    }

    /**
     * Sets the width (number of neurons) to use
     * @param hiddenWidth
     * @return the {@link Dense}
     */
    public Dense setHiddenWidth(int hiddenWidth) {
        this.hiddenWidth = hiddenWidth; return this;
    }

    /**
     * Sets the activation function
     * @param activation
     * @return the {@link Dense}
     */
    public Dense setActivation(IActivation activation) {
        this.activation = activation; return this;
    }
}
