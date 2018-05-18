package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Adds a layer with only an activation function
 *
 * @author Christian Skärby
 */
public class Act implements  LayerBlockConfig {

    private IActivation activation = new ActivationReLU();

    @Override
    public String name() {
        String actStr = LayerBlockConfig.actToStr(activation);
        return actStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        return builder
                .layer(info, new ActivationLayer.Builder()
                        .activation(activation)
                        .build());
    }

    /**
     * Sets activation function to use
     * @param activation
     * @return the {@link Act}
     */
    public Act setActivation(IActivation activation) {
        this.activation = activation;
        return this;
    }
}
