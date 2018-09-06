package ampcontrol.model.training.model.mutate;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
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


    static boolean doesNinPropagateToNext(GraphVertex vertex) {
        if(!vertex.hasLayer()) {
            return false;
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
}
