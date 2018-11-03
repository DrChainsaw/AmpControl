package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Traverses a Graph, recursively streaming all the children and grandchildren etc. of a given vertex.
 * @param <T> Type of the vertices of the graph
 *
 * @author Christian Skärby
 */
public class Traverse<T> implements Graph<T> {

    private final Graph<T> graph;
    private final Consumer<T> enterListener;
    private final Consumer<T> leaveListener;
    private final Predicate<T> traverse;

    public Traverse(Graph<T> graph){
        this(vertex -> true, vertex -> {}, vertex -> {}, graph);
    }

    public Traverse(Predicate<T> traverse, Graph<T> graph) {
        this(traverse, vertex -> {}, vertex -> {}, graph);
    }

    public Traverse(Predicate<T> traverse, Consumer<T> enterListener, Consumer<T> leaveListener, Graph<T> graph) {
        this.graph = graph;
        this.enterListener = enterListener;
        this.traverse = traverse;
        this.leaveListener = leaveListener;
    }

    @Override
    public Stream<T> children(T vertex) {
        enterListener.accept(vertex);
        final List<T> children = graph.children(vertex).collect(Collectors.toList());
        children.addAll(children.stream().filter(traverse)
                .flatMap(this::children).collect(Collectors.toList()));
        leaveListener.accept(vertex);
        return children.stream();
    }
}
