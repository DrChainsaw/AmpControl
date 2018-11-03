package ampcontrol.model.training.model.evolve.mutate.util;

import org.junit.Test;

import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Filter}
 *
 * @author Christian Skärby
 */
public class FilterTest {

    /**
     * Test that children are filtered
     */
    @Test
    public void children() {
        final String root = "root";
        final String pass = "pass";
        final String alsoPass = "alsoPass";
        final String block = "block";
        final String alsoBlock = "alsoBlock";

        assertEquals("", "pass, alsoPass",
                new Filter<>(vertex -> pass.equals(vertex) || alsoPass.equals(vertex), new MapGraph()
                        .addEdge(root, pass)
                        .addEdge(root, alsoPass)
                        .addEdge(root, block)
                        .addEdge(root, alsoBlock)
                        .asForward()).children(root).collect(Collectors.joining(", ")));
    }
}