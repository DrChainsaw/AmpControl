package ampcontrol.model.training.model.builder;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Overrides the name of a given {@link ModelBuilder}
 *
 * @author Christian Skärby
 */
public class OverrideName implements ModelBuilder {

    private final String name;
    private final ModelBuilder builder;

    public OverrideName(String name, ModelBuilder builder) {
        this.name = name;
        this.builder = builder;
    }

    @Override
    public MultiLayerNetwork build() {
        return builder.build();
    }

    @Override
    public ComputationGraph buildGraph() {
        return builder.buildGraph();
    }

    @Override
    public String name() {
        return name;
    }
}
