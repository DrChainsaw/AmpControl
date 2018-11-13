package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Traverses a Graph, recursively streaming all the children and grandchildren etc. of a given vertex.
 *
 * @param <T> Type of the vertices of the graph
 * @author Christian Skärby
 */
public class Traverse<T> implements Graph<T> {

    private final Graph<T> graph;
    private final Consumer<T> enterListener;
    private final Consumer<T> leaveListener;
    private final Predicate<T> traverse;

    public Traverse(Graph<T> graph) {
        this(vertex -> true, vertex -> {/* ignore */}, vertex -> {/* ignore */}, graph);
    }

    public Traverse(Predicate<T> traverse, Graph<T> graph) {
        this(traverse, vertex -> {/* ignore */}, vertex -> {/* ignore */}, graph);
    }

    /**
     * Returns a Graph which will only return leaf vertices
     * @param traverse Criterion for visiting vertices when traversing (only applies to children, not the "root" vertex)
     * @param graph {@link Graph} to traverse
     * @param <T> Type of Graph
     * @return a Graph
     */
    public static <T> Graph<T> leaves(Predicate<T> traverse, Graph<T> graph) {
        final Set<T> entered = new HashSet<>();
        return new Filter<>(vertex -> !entered.contains(vertex),
                new Traverse<>(
                        traverse,
                        entered::add,
                        vertex -> {/* Ignore*/},
                        graph));
    }

    /**
     * Constructor.
     *
     * @param traverse      Criterion for visiting vertices when traversing (only applies to children, not the "root" vertex)
     * @param enterListener Listener for whenever a vertex is entered before its children are visited. Note: Not invoked
     *                      through a Stream#peek call. Beware that the context for which it is valid only
     *                      refers to what happens when the underlying graphs child stream is consumed.
     * @param leaveListener Listener for whenever a vertex is left, meaning all its descendants have been visited. Note:
     *                      Not invoked through a Stream#peek call. Beware that the context for which it is valid only
     *                      refers to what happens when the underlying graphs child stream is consumed.
     * @param graph         {@link Graph} to traverse
     */
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
