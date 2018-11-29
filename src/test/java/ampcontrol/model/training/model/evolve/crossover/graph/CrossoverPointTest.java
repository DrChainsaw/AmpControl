package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
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