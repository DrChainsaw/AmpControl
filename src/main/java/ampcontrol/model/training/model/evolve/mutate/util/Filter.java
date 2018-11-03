package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Filters children according to a {@link Predicate}
 *
 * @param <T>
 * @author Christian Skärby
 */
public class Filter<T> implements Graph<T> {

    private final Predicate<T> predicate;
    private final Graph<T> graph;

    public Filter(Predicate<T> predicate, Graph<T> graph) {
        this.predicate = predicate;
        this.graph = graph;
    }

    @Override
    public Stream<T> children(T vertex) {
        return graph.children(vertex).filter(predicate);
    }
}
