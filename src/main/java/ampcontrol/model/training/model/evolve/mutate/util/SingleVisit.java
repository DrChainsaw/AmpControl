package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Filters out already visited vertices
 * @param <T>
 *
 * @author Christian Skärby
 */
public class SingleVisit<T> implements Graph<T> {

    private final Graph<T> graph;
    private final Set<T> visited = new HashSet<>();

    public SingleVisit(Graph<T> graph) {
        this.graph = graph;
    }

    @Override
    public Stream<T> children(T vertex) {
        return graph.children(vertex).filter(childVertex -> !visited.contains(childVertex)).peek(visited::add);
    }
}
