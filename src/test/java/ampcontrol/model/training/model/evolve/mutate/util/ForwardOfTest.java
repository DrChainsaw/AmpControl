package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ForwardOf}
 *
 * @author Christian Skärby
 */
public class ForwardOfTest {

    /**
     * Test that children returns the correct nodes
     */
    @Test
    public void childrenOfGraphBuilder() {
        final Graph<String> graph = new ForwardOf(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input1", "input2")
                .addVertex("vertex1_0", new ScaleVertex(1), "input1")
                .addVertex("vertex1_1", new ScaleVertex(2), "input1")
                .addVertex("vertex2_0", new ScaleVertex(3), "input2")
                .addVertex("vertex3_0", new ScaleVertex(4), "vertex1_0")
                .addVertex("vertex3_1", new ScaleVertex(4), "vertex1_0"));

        assertEquals("Incorrect children!",
                Arrays.asList("vertex1_0", "vertex1_1"), graph.children("input1").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Collections.singletonList("vertex2_0"), graph.children("input2").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Arrays.asList("vertex3_0", "vertex3_1"), graph.children("vertex1_0").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Collections.emptyList(), graph.children("vertex2_0").collect(Collectors.toList()));
    }

    /**
     * Test that children returns the correct nodes
     */
    @Test
    public void childrenOfGraphConfig() {
        final Graph<String> graph = new ForwardOf(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input1", "input2")
                .setOutputs("vertex3_1", "vertex3_0", "vertex2_0", "vertex1_1")
                .addVertex("vertex1_0", new ScaleVertex(1), "input1")
                .addVertex("vertex1_1", new ScaleVertex(2), "input1")
                .addVertex("vertex2_0", new ScaleVertex(3), "input2")
                .addVertex("vertex3_0", new ScaleVertex(4), "vertex1_0")
                .addVertex("vertex3_1", new ScaleVertex(4), "vertex1_0")
                .build());

        assertEquals("Incorrect children!",
                Arrays.asList("vertex1_0", "vertex1_1"), graph.children("input1").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Collections.singletonList("vertex2_0"), graph.children("input2").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Arrays.asList("vertex3_0", "vertex3_1"), graph.children("vertex1_0").collect(Collectors.toList()));

        assertEquals("Incorrect children!",
                Collections.emptyList(), graph.children("vertex2_0").collect(Collectors.toList()));

    }
}