package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.Map;
import java.util.stream.Stream;

/**
 * {@link Graph} in forward direction (input -> output) for a {@link ComputationGraphConfiguration}.
 *
 * @author Christian Skärby
 */
class ForwardOfConfig implements Graph<String> {

    private final ComputationGraphConfiguration config;

    ForwardOfConfig(ComputationGraphConfiguration config) {
        this.config = config;
    }

    @Override
    public Stream<String> children(String vertex) {
        return config.getVertexInputs().entrySet().stream()
                .filter(entry -> entry.getValue().contains(vertex))
                .map(Map.Entry::getKey);
    }
}
