package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.stream.Stream;

/**
 * Creates a {@link Graph<String>} in forward direction.
 *
 * @author Christian Skärby
 */
public class ForwardOf implements Graph<String> {

    private final Graph<String> actualGraph;

    public ForwardOf(ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        actualGraph = new ForwardOfBuilder(graphBuilder);
    }

    @Override
    public Stream<String> children(String vertex) {
        return actualGraph.children(vertex);
    }
}
