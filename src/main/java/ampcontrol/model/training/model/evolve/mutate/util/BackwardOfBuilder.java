package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.Collections;
import java.util.stream.Stream;

/**
 * {@link Graph} in backwards direction (output -> input) for a {@link ComputationGraphConfiguration.GraphBuilder}.
 *
 * @author Christian Skärby
 */
class BackwardOfBuilder implements Graph<String> {

    private final ComputationGraphConfiguration.GraphBuilder builder;

    BackwardOfBuilder(ComputationGraphConfiguration.GraphBuilder builder) {
        this.builder = builder;
    }

    @Override
    public Stream<String> children(String vertex) {
        return builder.getVertexInputs().getOrDefault(vertex, Collections.emptyList()).stream();
    }
}
