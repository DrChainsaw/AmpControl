package ampcontrol.model.training.model.evolve.mutate.util;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class TraverseTest {

    private final MapGraph mapGraph = new MapGraph()
            .addEdge("1", "1.1")
            .addEdge("1", "1.2")
            .addEdge("1.1", "1.1.1")
            .addEdge("1.1", "1.1.2")
            .addEdge("1.2", "1.2.1")
            .addEdge("1.2", "1.2.2")
            .addEdge("1.2.1", "1.2.1.1");

    @Test
    public void traverseAll() {
        assertEquals("Incorrect vertices!",
                "1.2.1, 1.2.2, 1.2.1.1",
                new Traverse<>(mapGraph.asForward()).children("1.2").collect(Collectors.joining(", ")));
    }

    /**
     * Expect that children with a length of more than 3 are not traversed.
     */
    @Test
    public void traverseShort() {
        assertEquals("Incorrect vertices!",
                "1.1, 1.2, 1.1.1, 1.1.2, 1.2.1, 1.2.2",
                new Traverse<>(
                        vertex -> vertex.length() < 4,
                        mapGraph.asForward()).children("1")
                        .collect(Collectors.joining(", ")));
    }

    /**
     * Test that enter and leave listeners are consistent with the graph
     */
    @Test
    public void traverseListen() {
        final List<String> expected = new ArrayList<>();
        final List<String> actual = new ArrayList<>();
        assertEquals("Incorrect vertices!",
                "1.1, 1.2, 1.1.1, 1.1.2, 1.2.1, 1.2.2, 1.2.1.1",
                new Traverse<>(
                        vertex -> true,
                        vertex -> expected.addAll(mapGraph.asForward().children(vertex).collect(Collectors.toList())),
                        vertex -> {
                            assertEquals("Incorrect vertices visited in vertex " + vertex + "!", expected, actual);
                            expected.removeAll(mapGraph.asForward().children(vertex).collect(Collectors.toList()));
                            actual.removeAll(mapGraph.asForward().children(vertex).collect(Collectors.toList()));
                        },
                        new Peek<>(actual::add,
                                mapGraph.asForward())).children("1")
                        .collect(Collectors.joining(", ")));
    }

}