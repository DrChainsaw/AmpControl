package ampcontrol.model.training.model.evolve.mutate;

import org.deeplearning4j.nn.graph.vertex.GraphVertex;

import java.util.stream.Stream;

/**
 * Interface for mutation operation.
 *
 * @param <T> Type to mutate
 * @author Christian Skärby
 */
public interface Mutation<T> {

    /**
     * Interface for supplying mutations. Motivation is to allow for serialization of mutation info as it might not
     * always be possible to create it from scratch.
     *
     * @param <T>
     */
    interface Supplier<T> {
        Stream<T> stream();
    }

    /**
     * Applies mutation to the provided input
     *
     * @param toMutate   The builder to mutate
     * @return The mutated builder. Note: might not be same instance as input!
     */
    T mutate(T toMutate);

    static boolean doesNinPropagateToNext(GraphVertex vertex) {
        if (!vertex.hasLayer()) {
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

    static boolean changeNinMeansChangeNout(GraphVertex vertex) {
        if (!vertex.hasLayer()) {
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
            case NORMALIZATION:
                return true;
            case SUBSAMPLING:
            case UPSAMPLING:
            case MULTILAYER:
            default:
                throw new UnsupportedOperationException("No idea what to do with this type: " + vertex.getLayer().type());

        }
    }
}
