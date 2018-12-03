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
 * {@link GraphVertex} which spies on Epsilons since it is not possible to obtain them through a listener. Note:
 *  Since vertexes must be deserializable the listeners must be added once the graph is built, most likely forcing
 *  a type cast to do so.
 *
 * @author Christian Skärby
 */
public class EpsilonSpyVertex extends GraphVertex {

    @Override
    public GraphVertex clone() {
        return new EpsilonSpyVertex();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof EpsilonSpyVertex;
    }

    @Override
    public int hashCode() {
        return 0;
    }

    @Override
    public int numParams(boolean backprop) {
        return 0;
    }

    @Override
    public int minVertexInputs() {
        return 1;
    }

    @Override
    public int maxVertexInputs() {
        return 1;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        return new EpsilonSpyVertexImpl(graph, name, idx);
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length != 1) {
            throw new InvalidInputTypeException("Can only handle one input! Got " + Arrays.toString(vertexInputs));
        }
        return vertexInputs[0];
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return new LayerMemoryReport.Builder(null, EpsilonSpyVertex.class, inputTypes[0], inputTypes[1])
                .standardMemory(0, 0).workingMemory(0, 0, 0, 0).cacheMemory(0, 0).build();
    }
}
