package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adds an {@link LSTM} layer
 *
 * @author Christian Skärby
 */
public class LstmBlock implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(LstmBlock.class);

    private int width = 64;
    private IActivation activation = new ActivationTanH();

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        actStr = actStr.isEmpty() ? actStr : "_" + actStr;
        return "L_" + width + actStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        log.info("LSTM layer " + info);
        BlockInfo nextInfo = builder.layer(
                info ,new LSTM.Builder()
                        .nOut(width)
                        .activation(activation)
                        .build());
        return new SimpleBlockInfo.Builder(nextInfo)
                .setPrevNrofOutputs(width)
                .build();
    }

    /**
     * Sets the width
     * @param width
     * @return the {@link LstmBlock}
     */
    public LstmBlock setWidth(int width) {
        this.width = width; return this;
    }

    /**
     * Sets the activation function to use
     * @param activation
     * @return the {@link LstmBlock}
     */
    public LstmBlock setActivation(IActivation activation) {
        this.activation = activation; return this;
    }
}
