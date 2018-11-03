package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Only returns children if a predicate is true
 * @param <T>
 *
 * @author Christian Skärby
 */
public class EnterIf<T> implements Graph<T> {

    private final Graph<T> graph;
    private final Predicate<T> predicate;

    public EnterIf(Predicate<T> predicate, Graph<T> graph) {
        this.graph = graph;
        this.predicate = predicate;
    }

    @Override
    public Stream<T> children(T vertex) {
        if(!predicate.test(vertex)) {
            return Stream.empty();
        }
        return graph.children(vertex);
    }
}
