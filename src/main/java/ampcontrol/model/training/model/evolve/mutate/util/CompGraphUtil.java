package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;

import java.util.Optional;
import java.util.function.Predicate;

/**
 * Various hideous utility functions around {@link ComputationGraph} to assist in traversal.
 */
public class CompGraphUtil {

    /**
     * Returns true if a change of size (NOut or NIn) propagates to the (next or prev) layers (NIn or NOut).
     * For example: If NOut is changed for conv1 in case conv1 -> maxpool -> conv2, then it is conv2 which
     * needs to change its NIn as the maxpool is transparent w.r.t sizes. The converse goes for the case
     * when NIn is changed in conv2; then conv1 needs to change NOut for the very same reason.
     *
     * @param computationGraph Configuration to check against
     * @return Predicate which is true if size change propagates
     */
    public static Predicate<String> changeSizePropagates(ComputationGraph computationGraph) {
        return vertex -> Optional.ofNullable(computationGraph.getVertex(vertex))
                .map(CompGraphUtil::doesSizeChangePropagate)
                .orElse(false);
    }

    private static boolean doesSizeChangePropagate(GraphVertex vertex) {
        if (!vertex.hasLayer()) {
            return true;
        }
        // Is there any parameter which can tell this instead of hardcoding it to types like this?
        switch (vertex.getLayer().type()) {
            case FEED_FORWARD:
            case RECURRENT:
            case CONVOLUTIONAL:
            case CONVOLUTIONAL3D:
            case RECURSIVE:
                return false;
            case SUBSAMPLING:
            case UPSAMPLING:
            case NORMALIZATION:
                return true;
            case MULTILAYER:
            default:
                throw new UnsupportedOperationException("No idea what to do with this type: " + vertex.getLayer().type());

        }
    }

    /**
     * Returns true if the given vertex requires that all its inputs have the same size. So far, only the
     * {@link ElementWiseVertex} is known to have this property.
     *
     * @param computationGraph Configuration to check against
     * @return Predicate which is true if size change propagates backwards
     */
    public static Predicate<String> changeSizePropagatesBackwards(ComputationGraph computationGraph) {
        return vertex -> Optional.ofNullable(computationGraph.getVertex(vertex))
                .map(CompGraphUtil::doesNOutChangePropagateToInputs)
                .orElse(false);
    }

    /**
     * Return true if a change in NOut propagates the NIn of all input layers.
     * Example of this is ElementWiseVertex: If one of the layers which is input
     * changes its NOut, the other input layer must also change its NOut
     *
     * @param vertex Vertex to check
     * @return True if change in NOut propagates to input layers
     */
    private static boolean doesNOutChangePropagateToInputs(GraphVertex vertex) {
        return vertex instanceof ElementWiseVertex;
    }

    /**
     * Creates a {@link ComputationGraphConfiguration.GraphBuilder} with the same config as a given {@link ComputationGraph}
     *
     * @param graph the {@link ComputationGraph}
     * @return the {@link ComputationGraphConfiguration.GraphBuilder}
     */
    public static ComputationGraphConfiguration.GraphBuilder toBuilder(ComputationGraph graph) {
        return new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration().clone(), new NeuralNetConfiguration.Builder(graph.conf().clone()));
    }
}
