package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.stream.Stream;

/**
 * Interface for generic graph. Main purpose is to hide some of the verbosity involved in finding vertices in graphs
 * which are not designed with easy traversal as a main requirement.
 *
 * @author Christian Skärby
 */
public interface Graph<T> {

    /**
     * Stream the children of the given vertex.
     * @param vertex Vertex to stream children from
     * @return Stream of child vertices.
     */
    Stream<T> children(T vertex);

}
