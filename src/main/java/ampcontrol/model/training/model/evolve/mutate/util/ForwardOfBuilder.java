package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.Map;
import java.util.stream.Stream;

/**
 * {@link Graph} in forward direction (input -> output) for a {@link ComputationGraphConfiguration.GraphBuilder}.
 *
 * @author Christian Skärby
 */
class ForwardOfBuilder implements Graph<String> {

    private final ComputationGraphConfiguration.GraphBuilder graphBuilder;

    public ForwardOfBuilder(ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        this.graphBuilder = graphBuilder;
    }

    @Override
    public Stream<String> children(String vertex) {
        return graphBuilder.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertex))
                .map(Map.Entry::getKey);
    }
}
