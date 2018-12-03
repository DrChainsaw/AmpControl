package ampcontrol.model.training.model.builder;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Interface for building models.
 *
 * @author Christian Skärby
 */
public interface ModelBuilder {

    /**
     * Construct a {@link MultiLayerNetwork} from the builder.
     *
     * @return a {@link MultiLayerNetwork}
     */
    MultiLayerNetwork build();

    /**
     * Construct a {@link ComputationGraph} from the builder.
     * @return a {@link ComputationGraph}
     */
    ComputationGraph buildGraph();

    /**
     * Returns the name of the model the builder will create.
     *
     * @return the name of the model the builder will create.
     */
    String name();
}
