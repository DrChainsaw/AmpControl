package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.function.LongSupplier;
import java.util.stream.Stream;

/**
 * Limits the number of children returned from a {@link Graph}
 * @param <T>
 *
 * @author Christian Skärby
 */
public class Limit<T> implements Graph<T> {

    private final LongSupplier limit;
    private final Graph<T> graph;

    public Limit(long limit, Graph<T> graph) {
        this(() -> limit, graph);
    }

    public Limit(LongSupplier limit, Graph<T> graph) {
        this.limit = limit;
        this.graph = graph;
    }

    @Override
    public Stream<T> children(T vertex) {
        return graph.children(vertex).limit(limit.getAsLong());
    }
}
