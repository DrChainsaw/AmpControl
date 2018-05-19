package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Sets {@link InputType} of whole model to be convolutional 1D. Assumes dimension 0 of input to be time while dimension
 * 1 is frequency so that convolution is done in frequency domain.
 *
 * @author Christian Skärby
 */
public class ConvTimeType implements LayerBlockConfig {

    private final int[] inputShape;

    public ConvTimeType(int[] inputShape) {
        this.inputShape = inputShape;
        if(inputShape.length != 2) {
            throw new IllegalArgumentException("Input must be length 2!");
        }
    }

    @Override
    public String name() {
        return "";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new UnsupportedOperationException("Not supported!");
    }

    @Override
    public BlockInfo addLayers(NeuralNetConfiguration.ListBuilder listBuilder, BlockInfo info) {
        listBuilder
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.recurrent(inputShape[0], inputShape[1]));
        listBuilder.setBackpropType(BackpropType.Standard);
        return info;
    }

    @Override
    public BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, BlockInfo info) {
        graphBuilder
                .backprop(true)
                .pretrain(false)
                .setInputTypes(InputType.recurrent(inputShape[0], inputShape[1]));
        graphBuilder.setBackpropType(BackpropType.Standard);
        return info;
    }
}
