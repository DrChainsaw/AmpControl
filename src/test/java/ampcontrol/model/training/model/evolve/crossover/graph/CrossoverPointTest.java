package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.Traverse;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link CrossoverPoint}
 *
 * @author Christian Skärby
 */
public class CrossoverPointTest {

    /**
     * Test to do crossover when the vertex name is identical
     */
    @Test
    public void executeSameVertexName() {
        final ComputationGraphConfiguration.GraphBuilder builder1 = CompGraphUtil.toBuilder(GraphUtils.getGraph("0", "1", "2"))
                .setInputTypes(InputType.feedForward(33));

        final ComputationGraphConfiguration.GraphBuilder builder2 = CompGraphUtil.toBuilder(GraphUtils.getGraph("0", "1", "2"))
                .setInputTypes(InputType.feedForward(33));

        GraphInfo result = new CrossoverPoint(
                new VertexData("1", new GraphInfo.Input(builder1)),
                new VertexData("1", new GraphInfo.Input(builder2)))
                .execute();

        final ComputationGraph graph = new ComputationGraph(result.builder().build());
        graph.init();
        graph.output(Nd4j.randn(new long[] {1, 33}));
    }

    /**
     * Test to do crossover twice
     */
    @Test
    public void executeCrossTwice() {
        final ComputationGraph graph = GraphUtils.getGraph("0", "1", "2");
        final ComputationGraphConfiguration.GraphBuilder builder = CompGraphUtil.toBuilder(graph)
                .setInputTypes(InputType.feedForward(33));

        final GraphInfo input1 = new GraphInfo.Input(builder);
        final GraphInfo input2 = new GraphInfo.Input(builder);
        GraphInfo result = new CrossoverPoint(
                new VertexData("1", input1),
                new VertexData("1", input2))
                .execute();

        final ComputationGraph newGraph = new ComputationGraph(result.builder().build());
        newGraph.init();

        final GraphInfo input11 = new GraphInfo.Input(result.builder());
        final GraphInfo input22 = new GraphInfo.Input(result.builder());
        GraphInfo result1 = new CrossoverPoint(
                new VertexData("1", input11),
                new VertexData("1", input22))
                .execute();

        final List<String> allNames =
                Stream.concat(
                        result1.verticesFrom(input11)
                                .map(GraphInfo.NameMapping::getNewName),
                        result1.verticesFrom(input22)
                                .map(GraphInfo.NameMapping::getNewName))
                        .collect(Collectors.toList());

        final List<String> expectedNames = new Traverse<>(new ForwardOf(result1.builder())).children("input")
                .collect(Collectors.toList());

        assertEquals("Incorrect names!", expectedNames, allNames);

        final ComputationGraph newNewGraph = new ComputationGraph(result1.builder().build());
        newNewGraph.init();
        newNewGraph.output(Nd4j.randn(new long[] {1, 33}));
    }

    /**
     * Test to do when the top graph already contains unique names which happens to be identical to those generated when
     * trying to come up with a new unique name for the first vertex after the crossover point
     */
    @Test
    public void executeWithTopDuplicateName() {
        final ComputationGraphConfiguration.GraphBuilder builder1 = CompGraphUtil.toBuilder(GraphUtils.getGraph("0", "1", "2"))
                .setInputTypes(InputType.feedForward(33));

        final ComputationGraphConfiguration.GraphBuilder builder2 = CompGraphUtil.toBuilder(GraphUtils.getGraph("-1", "0", "0_0"))
                .setInputTypes(InputType.feedForward(33));

        final GraphInfo input1 = new GraphInfo.Input(builder1);
        final GraphInfo input2 = new GraphInfo.Input(builder2);
        GraphInfo result = new CrossoverPoint(
                new VertexData("0", input1),
                new VertexData("-1", input2))
                .execute();

        final Set<String> allNames =
                Stream.concat(
                        result.verticesFrom(input1)
                                .map(GraphInfo.NameMapping::getNewName),
                        result.verticesFrom(input2)
                                .map(GraphInfo.NameMapping::getNewName))
                        .collect(Collectors.toSet());

        final Set<String> expectedNames = new Traverse<>(new ForwardOf(result.builder())).children("input")
                .collect(Collectors.toSet());

        assertEquals("Incorrect names!", expectedNames, allNames);

        final ComputationGraph newNewGraph = new ComputationGraph(result.builder().build());
        newNewGraph.init();
        newNewGraph.output(Nd4j.randn(new long[] {1, 33}));

    }

    /**
     * Test to do crossover of a vertex before an ElementWiseVertex.
     */
    @Test
    public void executeBeforeElemwiseAdd() {
        final ComputationGraphConfiguration.GraphBuilder builder1 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("0", new Convolution2D.Builder(2,2).nOut(4).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("1", new Convolution2D.Builder(2,2).nOut(4).convolutionMode(ConvolutionMode.Same).build(), "0")
                .addVertex("add0And1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "0", "1")
                .addLayer("2", new BatchNormalization(), "add0And1")
                .addLayer("output", new CnnLossLayer(), "2")
                .setInputTypes(InputType.convolutional(33,33,3));

        final ComputationGraphConfiguration.GraphBuilder builder2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output")
                .addLayer("0", new Convolution2D.Builder(2,2).nOut(7).convolutionMode(ConvolutionMode.Same).build(), "input")
                .addLayer("1", new Convolution2D.Builder(2,2).nOut(9).convolutionMode(ConvolutionMode.Same).build(), "0")
                .addLayer("output", new CnnLossLayer(), "1")
                .setInputTypes(InputType.convolutional(33,33,3));

        builder1.getLayerActivationTypes();
        builder2.getLayerActivationTypes();

        GraphInfo result = new CrossoverPoint(
                new VertexData("2", new GraphInfo.Input(builder1)),
                new VertexData("1", new GraphInfo.Input(builder2)))
                .execute();

        final ComputationGraph graph = new ComputationGraph(result.builder().build());
        graph.init();
        graph.output(Nd4j.randn(new long[] {1, 3, 33, 33}));
    }
}