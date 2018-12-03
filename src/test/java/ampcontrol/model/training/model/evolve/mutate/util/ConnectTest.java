package ampcontrol.model.training.model.evolve.mutate.util;

import org.junit.Test;

import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Connect}
 *
 * @author Christian Skärby
 */
public class ConnectTest {

    /**
     * Test that children of one graph is used as parents for the next graph
     */
    @Test
    public void children() {
        MapGraph graph = new MapGraph()
                .addEdge("root", "1")
                .addEdge("root", "2")
                .addEdge("1", "1.1")
                .addEdge("1", "1.2")
                .addEdge("2", "2.1");

        assertEquals("Incorrect output!", "1, 1",
                new Connect<>(graph.asForward(), graph.asBackward()).children("1")
                        .collect(Collectors.joining(", ")));
    }

    /**
     * Test with an empty graph. Result basically depends on graph implementation, but at least it is tested that Connect
     * can handle empty graphs
     */
    @Test
    public void empty() {
        MapGraph graph = new MapGraph();
        assertEquals("Incorrect output!", "",
                new Connect<>(graph.asForward(), graph.asBackward()).children("")
                        .collect(Collectors.joining(", ")));
    }

}