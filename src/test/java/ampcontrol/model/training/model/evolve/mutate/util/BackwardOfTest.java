package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link BackwardOf}
 *
 * @author Christian Skärby
 */
public class BackwardOfTest {

    /**
     * Test that children returns the correct nodes
     */
    @Test
    public void childrenOfGraphBuilder() {
        final Graph<String> graph = new BackwardOf(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input1", "input2")
                .addVertex("vertex1_0", new ScaleVertex(1), "input1")
                .addVertex("vertex2_0", new MergeVertex(), "vertex1_0", "input2"));

        assertEquals("Incorrect children!",
                Collections.emptyList(), graph.children("input1").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Collections.singletonList("input1"), graph.children("vertex1_0").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Arrays.asList("vertex1_0", "input2"), graph.children("vertex2_0").collect(Collectors.toList()));
    }
}