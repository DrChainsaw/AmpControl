package ampcontrol.model.training.model.evolve.mutate.util;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Peek}
 *
 * @author Christian Skärby
 */
public class PeekTest {

    /**
     * Test that children are peeked
     */
    @Test
    public void children() {
        final MapGraph mapGraph = new MapGraph();
        final List<String> expected = IntStream.range(0,4)
                .mapToObj(i -> "child" + i)
                .peek(child -> mapGraph.addEdge("root", child))
                .collect(Collectors.toList());
        final List<String> actual = new ArrayList<>();
        final Graph<String> graph = new Peek<>(actual::add, mapGraph.asForward());

        assertEquals("Incorrect children returned!", expected, graph.children("root").collect(Collectors.toList()));
        assertEquals("Children were not peeked!", expected, actual);
    }
}