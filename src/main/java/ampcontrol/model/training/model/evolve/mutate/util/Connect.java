package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.stream.Stream;

/**
 * Connects two graphs, so that the children of the first graph is used to query the second graph. Main use case is to
 * connect {@link ForwardOf} with {@link BackwardOf} or vice versa and {@link Traverse}.
 * @param <T>
 * @author Christian Skärby
 */
public class Connect<T> implements Graph<T> {

    private final Graph<T> first;
    private final Graph<T> second;

    public Connect(Graph<T> first, Graph<T> second) {
        this.first = first;
        this.second = second;
    }


    @Override
    public Stream<T> children(T vertex) {
        return first.children(vertex).flatMap(second::children);
    }
}
