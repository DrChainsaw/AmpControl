package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Sets {@link InputType} to RNN. Assumes dimension 0 of input is time and dimension 1 is the number of time sequences,
 * e.g. 1 for a pure time series or number of frequency bins for spectrogram input.
 *
 * @author Christian Skärby
 */
public class RnnType implements LayerBlockConfig {

    private final int[] inputShape;
    private int tbpttLength = -1;

    public RnnType(int[] inputShape) {
        this.inputShape = inputShape;
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
                .setInputType(InputType.recurrent(inputShape[1], inputShape[0]));
        if(tbpttLength > 0) {
            listBuilder.setBackpropType(BackpropType.TruncatedBPTT);
            listBuilder.setTbpttBackLength(tbpttLength);
            listBuilder.setTbpttFwdLength(tbpttLength);
        } else {
            listBuilder.setBackpropType(BackpropType.Standard);
        }
        return info;
    }

    @Override
    public BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, BlockInfo info) {
        graphBuilder
                .backprop(true)
                .pretrain(false)
                .setInputTypes(InputType.recurrent(inputShape[1], inputShape[0]));

        if(tbpttLength > 0) {
            graphBuilder.setBackpropType(BackpropType.TruncatedBPTT);
            graphBuilder.setTbpttBackLength(tbpttLength);
            graphBuilder.setTbpttFwdLength(tbpttLength);
        } else {
            graphBuilder.setBackpropType(BackpropType.Standard);
        }
        return info;
    }

    /**
     * Sets lenght of truncated backpropagation through time
     * @param tbpttLength
     * @return the {@link RnnType}
     */
    public RnnType setTbpttLength(int tbpttLength) {
        this.tbpttLength = tbpttLength;
        return this;
    }
}
