package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.stream.Stream;

/**
 * Creates a {@link Graph<String>} in backward (output -> input) direction.
 *
 * @author Christian Skärby
 */
public class BackwardOf implements Graph<String> {

    private final Graph<String> actualGraph;

    public BackwardOf(ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        actualGraph = new BackwardOfBuilder(graphBuilder);
    }

    public BackwardOf(ComputationGraph computationGraph) {
        actualGraph = new BackwardOfCompGraph(computationGraph);
    }

    @Override
    public Stream<String> children(String vertex) {
        return actualGraph.children(vertex);
    }

}
