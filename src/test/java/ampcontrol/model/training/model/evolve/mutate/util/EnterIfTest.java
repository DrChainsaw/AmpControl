package ampcontrol.model.training.model.evolve.mutate.util;

import org.junit.Test;

import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link EnterIf}
 *
 * @author Christian Skärby
 */
public class EnterIfTest {

    /**
     * Test that vertices are filtered as expected
     */
    @Test
    public void children() {
        final String block = "block";
        final String pass = "pass";
        final Graph<String> graph = new EnterIf<>(pass::equals, new MapGraph()
        .addEdge(block, pass)
        .addEdge(pass, block)
        .asForward());

        assertEquals("Expected to be blocked!", "", graph.children(block).collect(Collectors.joining()));
        assertEquals("Did not expect to beblocked!", block, graph.children(pass).collect(Collectors.joining()));
    }
}