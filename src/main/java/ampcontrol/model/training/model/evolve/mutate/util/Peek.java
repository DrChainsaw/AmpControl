package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * Peeks the stream of children
 * @param <T>
 * @author Christian Skärby
 */
public class Peek<T> implements Graph<T> {

    private final Consumer<T> consumer;
    private final Graph<T> graph;

    public Peek(Consumer<T> consumer, Graph<T> graph) {
        this.consumer = consumer;
        this.graph = graph;
    }

    @Override
    public Stream<T> children(T vertex) {
        return graph.children(vertex).peek(consumer);
    }
}
