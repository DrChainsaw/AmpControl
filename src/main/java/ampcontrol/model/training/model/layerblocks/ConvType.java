package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Sets {@link InputType} to convolutional 2D.
 *
 * @author Christian Skärby
 */
public class ConvType implements LayerBlockConfig {

    private final int[] inputShape;

    public ConvType(int[] inputShape) {
        this.inputShape = inputShape;
    }

    @Override
    public String name() {
        return "";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {throw new UnsupportedOperationException("Not supported!");}

    @Override
    public BlockInfo addLayers(NeuralNetConfiguration.ListBuilder listBuilder, BlockInfo info) {
        listBuilder
                .setInputType(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]));
        return info;
    }

    @Override
    public BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, BlockInfo info) {
        graphBuilder
                .setInputTypes(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]));
        return info;
    }
}
