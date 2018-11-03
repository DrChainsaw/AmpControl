package ampcontrol.model.training.model.evolve.mutate.util;

import java.util.*;

/**
 * Simple graph for testing purposes
 *
 * @author Christian Skärby
 */
class MapGraph {
    private final Map<String, List<String>> graph = new HashMap<>();
    private final Map<String, List<String>> reversegraph = new HashMap<>();

    MapGraph addEdge(String parent, String child) {
        graph.computeIfAbsent(parent, key -> new ArrayList<>()).add(child);
        reversegraph.computeIfAbsent(child, key -> new ArrayList<>()).add(parent);
        return this;
    }

    Graph<String> asForward() {
        return vertex -> graph.getOrDefault(vertex, Collections.emptyList()).stream();
    }

    Graph<String> asBackward() {
        return vertex -> reversegraph.getOrDefault(vertex, Collections.emptyList()).stream();
    }
}
