package ampcontrol.model.training.model.evolve.mutate.util;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link SingleVisit}
 *
 * @author Christian Skärby
 */
public class SingleVisitTest {

    /**
     * Test that children are only visited once
     */
    @Test
    public void children() {
        final Graph<String> graph = new SingleVisit<>(new MapGraph()
                .addEdge("1", "1.1")
                .addEdge("1", "1.2")
                .addEdge("1.2", "1.2.1")
                .addEdge("1.2", "1.2.1")
        .asForward());

        assertNotEquals("Incorrect number of children returned!", 0, graph.children("1").count());
        assertEquals("Expected children to not be visited again!", 0, graph.children("1").count());

        assertNotEquals("Incorrect number of children returned!", 0, graph.children("1.2").count());
        assertEquals("Expected children to not be visited again!", 0, graph.children("1.2").count());
    }
}