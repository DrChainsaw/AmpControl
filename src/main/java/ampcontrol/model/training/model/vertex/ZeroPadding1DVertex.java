package ampcontrol.model.training.model.vertex;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Ripped out of dl4j master and converted to a vertex as I couldn't figure out how to do custom layers.
 * TODO: Remove when upgrading dl4j to 9.2-xxx
 */
public class ZeroPadding1DVertex extends GraphVertex {

    public ZeroPadding1DVertex(int[] padding) {
        this.padding = padding;
    }

    private final int[] padding;

    @Override
    public ZeroPadding1DVertex clone() {
        return new ZeroPadding1DVertex(padding);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ZeroPadding1DVertex))
            return false;
        return Arrays.equals(padding, ((ZeroPadding1DVertex) o).padding);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(padding);
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
        return new ZeroPadding1DVertexImpl(graph, name, idx, padding);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        InputType first = vertexInputs[0];

        int size = 0;
        if (first instanceof InputType.InputTypeRecurrent) {
            size = ((InputType.InputTypeRecurrent) first).getSize();
        } else {
            throw new InvalidInputTypeException(
                    "Invalid input: Zeropad vertex 1 " + this.toString() + "wrong inputtype!"
                            + "Must be RNN type! Was " + first);
        }

        InputType output = new InputType.InputTypeRecurrent(size + Arrays.stream(padding).sum(),
                ((InputType.InputTypeRecurrent) first).getTimeSeriesLength());
        return first;
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        //No working memory in addition to output activations
        return new LayerMemoryReport.Builder(null, ampcontrol.model.training.model.vertex.ChannelMultVertex.class, inputTypes[0], inputTypes[0])
                .standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();
    }
}
