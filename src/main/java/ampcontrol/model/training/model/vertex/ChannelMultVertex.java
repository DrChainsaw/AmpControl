package ampcontrol.model.training.model.vertex;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * {@link GraphVertex} which multiplies each channel in a convolutional activation of size [b,c,h,w] with a scalar of
 * size (b,c) where b is the batch size, c is the number of channels, h is the height and w is the width. Used in
 * squeeze-exitation networks: https://arxiv.org/abs/1709.01507
 *
 * @author Christian Skärby
 */
public class ChannelMultVertex extends GraphVertex {

    @Override
    public ChannelMultVertex clone() {
        return new ChannelMultVertex();
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ChannelMultVertex))
            return false;
        return true; //??
    }

    @Override
    public int hashCode() {
        return ChannelMultVertex.class.hashCode();
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 2;
    }

    @Override
    public int maxVertexInputs() {
        return 2;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        return new ChannelMultVertexImpl(graph, name, idx);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        InputType first = vertexInputs[0];

        long numChannels = 0;
        if (first instanceof InputType.InputTypeConvolutional) {
            numChannels = ((InputType.InputTypeConvolutional) first).getChannels();
        } else if (first instanceof InputType.InputTypeConvolutionalFlat) {
            numChannels = ((InputType.InputTypeConvolutionalFlat) first).getDepth();
        }
        boolean ok = false;
        InputType second = vertexInputs[1];
        if (second instanceof InputType.InputTypeFeedForward) {
            ok = ((InputType.InputTypeFeedForward) second).getSize() == numChannels;
        }

        if (!ok) {
            throw new InvalidInputTypeException(
                    "Invalid input: Channel mult vertex " + this.toString() + "size mismatch! "
                            + "Depth of first type (" + first + ") must be equal to size of second type ("
                             + second + ")!");
        }

        return first;
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, ChannelMultVertex.class, inputTypes[0], inputTypes[1])
                .standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();
    }

}
