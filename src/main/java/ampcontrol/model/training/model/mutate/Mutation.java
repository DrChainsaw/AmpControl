package ampcontrol.model.training.model.mutate;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;

/**
 * Interface for mutation operation.
 *
 * @author Christian Skärby
 */
public interface Mutation {
    /**
     * Applies mutation to the provided {@link TransferLearning.GraphBuilder}.
     * @param builder The builder to mutate
     * @param prevGraph The previous graph
     * @return The mutated builder
     */
    TransferLearning.GraphBuilder mutate(TransferLearning.GraphBuilder builder, ComputationGraph prevGraph);
}
