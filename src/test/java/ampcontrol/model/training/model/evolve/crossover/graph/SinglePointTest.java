package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

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

        final InputType inputType = InputType.convolutional(33, 33, 3);
        final GraphInfo info1 = inputOf(graph1, inputType);
        final GraphInfo info2 = inputOf(graph2, inputType);


        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(-0.125, 0d)).cross(info1, info2);

        final Collection<String> expected = Arrays.asList("first1", "second2", "second3");
        final Collection<String> notExpected = Arrays.asList("second1", "first2", "first3");

        assertTrue("Expected vertices not present!",
                output.builder().getVertices().keySet().containsAll(expected));
        assertFalse("Not expected vertices are present!",
                output.builder().getVertices().keySet().stream().anyMatch(notExpected::contains));

        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info1).anyMatch(vertex -> vertex.getNewName().matches("first1")));
        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info1).noneMatch(vertex -> vertex.getNewName().matches("second.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info1)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex.getNewName())));

        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info2).anyMatch(vertex -> vertex.getNewName().matches("second[1,2]")));
        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info2).noneMatch(vertex -> vertex.getNewName().matches("first.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info2)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex.getNewName())));

        assertEquals("Not all vertices are mapped!", output.builder().getVertices().keySet(),
                Stream.concat(output.verticesFrom(info1), output.verticesFrom(info2))
                        .map(GraphInfo.NameMapping::getNewName)
                        .collect(Collectors.toSet()));

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

        final InputType inputType = InputType.convolutional(33, 33, 3);
        final GraphInfo info1 = inputOf(graph1, inputType);
        final GraphInfo info2 = inputOf(graph2, inputType);


        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(-0.7, 0.1)).cross(info1, info2);

        final Collection<String> expected = Arrays.asList("firstBefore", "secondAfter");
        final Collection<String> notExpected = Arrays.asList("secondBefore", "firstAfter", "first1", "second1");

        assertTrue("Expected vertices not present!",
                output.builder().getVertices().keySet().containsAll(expected));
        assertFalse("Not expected vertices are present!",
                output.builder().getVertices().keySet().stream().anyMatch(notExpected::contains));

        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info1).anyMatch(vertex -> vertex.getNewName().matches("firstBefore")));
        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info1).noneMatch(vertex -> vertex.getNewName().matches("second.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info1)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex.getNewName())));

        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info2).anyMatch(vertex -> vertex.getNewName().matches("secondAfter")));
        assertTrue("Vertices not mapped to correct info!",
                output.verticesFrom(info2).noneMatch(vertex -> vertex.getNewName().matches("first.*")));
        assertFalse("Vertices not mapped to correct info!", output.verticesFrom(info2)
                .anyMatch(vertex -> !output.builder().getVertices().containsKey(vertex.getNewName())));

        assertEquals("Not all vertices are mapped!", output.builder().getVertices().keySet(),
                Stream.concat(output.verticesFrom(info1), output.verticesFrom(info2))
                        .map(GraphInfo.NameMapping::getNewName)
                        .collect(Collectors.toSet()));

        final ComputationGraph newGraph = new ComputationGraph(output.builder().build());
        newGraph.init();

        long[] shape = inputType.getShape(true);
        shape[0] = 1;
        newGraph.outputSingle(Nd4j.randn(shape));
    }

    /**
     * Test that crossoverpoint is not an {@link EpsilonSpyVertex} as they are typically inserted after very specific layers
     */
    @Test
    public void avoidSizeTooSmall() {
        final InputType inputType1 = InputType.convolutional(3, 3, 2);

        final ComputationGraphConfiguration.GraphBuilder builder1 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(2,2).nOut(1).build(), "input")
                .addLayer("output", new CnnLossLayer(), "conv")
                .setInputTypes(inputType1);

        final InputType inputType2 = InputType.convolutional(6, 6, 2);
        final ComputationGraphConfiguration.GraphBuilder builder2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(4,4).nOut(1).build(), "input")
                .addLayer("output", new CnnLossLayer(), "conv")
                .setInputTypes(inputType2);

        final GraphInfo info1 = new GraphInfo.Input(builder1);
        final GraphInfo info2 = new GraphInfo.Input(builder2);

        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(0.0, 0)).cross(info1, info2);

        assertEquals("Expected crossover to be noop!", builder1, output.builder());

        final ComputationGraph graph = new ComputationGraph(output.builder().build());
        graph.init();
        graph.output(Nd4j.randn(new long[] {1,2,3,3}));
    }

    /**
     * Test that crossoverpoint is not an {@link EpsilonSpyVertex} as they are typically inserted after very specific layers
     */
    @Test
    public void avoidSpyVertexInTop() {
        final InputType inputType = InputType.convolutional(4, 4, 2);

        final ComputationGraphConfiguration.GraphBuilder builder1 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(2,2).build(), "input")
                .addLayer("batchNorm", new BatchNormalization(), "conv")
                .addLayer("output", new CnnLossLayer(), "batchNorm")
                .setInputTypes(inputType);

        final ComputationGraphConfiguration.GraphBuilder builder2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(2,2).build(), "input")
                .addVertex("convSpy", new EpsilonSpyVertex(), "conv")
                .addLayer("output", new CnnLossLayer(), "convSpy")
                .setInputTypes(inputType);

        final GraphInfo info1 = new GraphInfo.Input(builder1);
        final GraphInfo info2 = new GraphInfo.Input(builder2);

        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(-0.0, 2/3d)).cross(info1, info2);

        new ForwardOf(output.builder()).children("batchNorm")
                .map(childName -> output.builder().getVertices().get(childName))
                .forEach(
                childVertex -> assertNotEquals("Batchnorm must not be input to spy vertex!!",
                        EpsilonSpyVertex.class, childVertex.getClass()));
    }

    /**
     * Test that crossoverpoint is not before an {@link EpsilonSpyVertex} as they are typically inserted after very
     * specific layers.
     */
    @Test
    public void avoidSpyVertexInBottom() {
        final InputType inputType = InputType.convolutional(4, 4, 2);

        final ComputationGraphConfiguration.GraphBuilder builder1 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(2,2).build(), "input")
                .addVertex("convSpy", new EpsilonSpyVertex(), "conv")
                .addLayer("output", new CnnLossLayer(), "convSpy")
                .setInputTypes(inputType);

        final ComputationGraphConfiguration.GraphBuilder builder2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(2,2).build(), "input")
                .addLayer("batchNorm", new BatchNormalization(), "conv")
                .addLayer("output", new CnnLossLayer(), "batchNorm")
                .setInputTypes(inputType);


        final GraphInfo info1 = new GraphInfo.Input(builder1);
        final GraphInfo info2 = new GraphInfo.Input(builder2);

        // Conv would lose its eps spy unless SinglePoint takes some action to prevent this
        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(-0.0, 1/3d)).cross(info1, info2);

        assertTrue("Conv must still have spy as output!", new ForwardOf(output.builder()).children("conv")
                .map(childName -> output.builder().getVertices().get(childName))
                .anyMatch(
                        childVertex -> childVertex instanceof EpsilonSpyVertex));
    }

    /**
     * Test that crossoverpoint is not before an {@link EpsilonSpyVertex} as they are typically inserted after very
     * specific layers.
     */
    @Test
    public void avoidGlobalPool() {
        final InputType inputType = InputType.convolutional(4, 4, 2);

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("conv", new Convolution2D.Builder(2,2).nOut(1).build(), "input")
                .addLayer("gp", new GlobalPoolingLayer(), "conv")
                .addLayer("dense", new DenseLayer.Builder().nOut(1).build(), "gp")
                .addLayer("output", new OutputLayer.Builder().nOut(1).build(), "dense")
                .setInputTypes(inputType);


        final GraphInfo info1 = new GraphInfo.Input(builder);
        final GraphInfo info2 = new GraphInfo.Input(builder);

        // This will cause SinglePoint to connect a dense layer to a global pooling layer -> crash!
        final GraphInfo output = new SinglePoint(() -> new SinglePoint.PointSelection(0.2, 3/4d)).cross(info1, info2);

        final ComputationGraph graph = new ComputationGraph(output.builder().build());
        graph.init();
        graph.output(Nd4j.randn(new long[] {1,2,4,4}));
    }

    private static GraphInfo inputOf(ComputationGraph graph, InputType inputType) {
        return new GraphInfo.Input(
                CompGraphUtil.toBuilder(graph)
                        .setInputTypes(inputType));
    }
}