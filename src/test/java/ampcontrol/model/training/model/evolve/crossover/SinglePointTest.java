package ampcontrol.model.training.model.evolve.crossover;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.crossover.graph.GraphInfo;
import ampcontrol.model.training.model.evolve.crossover.graph.SinglePoint;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link SinglePoint}
 *
 * @author Christian Skärby
 */
public class SinglePointTest {

    /**
     * Test crossover between two simple convolutional graphs.
     */
    @Test
    public void crossSimpleConv() {
        final ComputationGraph graph1 = GraphUtils.getCnnGraph("first1", "first2", "first3");
        final ComputationGraph graph2 = GraphUtils.getCnnGraph("second1", "second2", "second3");

        final InputType inputType = InputType.convolutional(33,33,3);
        final GraphInfo info1 =  inputOf(graph1, inputType);
        final GraphInfo info2 =  inputOf(graph2, inputType);


        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(-0.125, 0d)).cross(info1, info2);

        final Collection<String> expected = Arrays.asList("first1", "second2", "second3");
        final Collection<String> notExpected = Arrays.asList("second1", "first2", "first3");

        assertTrue("Expected vertices not present!",
                output.builder().getVertices().keySet().containsAll(expected));
        assertFalse("Not expected vertices are present!",
                output.builder().getVertices().keySet().stream().anyMatch(notExpected::contains));

        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info1).anyMatch(vertex -> vertex.matches("first1")));
        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info1).noneMatch(vertex -> vertex.matches("second.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info1)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex)));

        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info2).anyMatch(vertex -> vertex.matches("second[1,2]")));
        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info2).noneMatch(vertex -> vertex.matches("first.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info2)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex)));

        assertEquals("Not all vertices are mapped!", output.builder().getVertices().keySet(),
                Stream.concat(output.verticesFrom(info1), output.verticesFrom(info2)).collect(Collectors.toSet()));

        final ComputationGraph newGraph = new ComputationGraph(output.builder().build());
        newGraph.init();

        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        newGraph.outputSingle(Nd4j.randn(shape));
    }

    /**
     * Test crossover between one forked graph and one forked residual graph.
     */
    @Test
    public void crossForkConv() {
        final ComputationGraph graph1 = GraphUtils.getForkNet("firstBefore", "firstAfter", "first1", "first2", "first3");
        final ComputationGraph graph2 = GraphUtils.getForkResNet("secondBefore", "secondAfter", "second1", "second2", "second3");

        final InputType inputType = InputType.convolutional(33,33,3);
        final GraphInfo info1 =  inputOf(graph1, inputType);
        final GraphInfo info2 =  inputOf(graph2, inputType);


        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(-0.7, 0.1)).cross(info1, info2);

        final Collection<String> expected = Arrays.asList("firstBefore", "secondAfter");
        final Collection<String> notExpected = Arrays.asList("secondBefore", "firstAfter", "first1", "second1");

        assertTrue("Expected vertices not present!",
                output.builder().getVertices().keySet().containsAll(expected));
        assertFalse("Not expected vertices are present!",
                output.builder().getVertices().keySet().stream().anyMatch(notExpected::contains));

        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info1).anyMatch(vertex -> vertex.matches("firstBefore")));
        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info1).noneMatch(vertex -> vertex.matches("second.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info1)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex)));

        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info2).anyMatch(vertex -> vertex.matches("secondAfter")));
        assertTrue("Vertices not mapped to correct info!", output.verticesFrom(info2).noneMatch(vertex -> vertex.matches("first.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info2)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex)));

        assertEquals("Not all vertices are mapped!", output.builder().getVertices().keySet(),
                Stream.concat(output.verticesFrom(info1), output.verticesFrom(info2)).collect(Collectors.toSet()));

        final ComputationGraph newGraph = new ComputationGraph(output.builder().build());
        newGraph.init();

        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        newGraph.outputSingle(Nd4j.randn(shape));
    }

    private static GraphInfo inputOf(ComputationGraph graph, InputType inputType) {
        return new GraphInfo.Input(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(),
                        new NeuralNetConfiguration.Builder(graph.conf()))
        .setInputTypes(inputType));
    }
}