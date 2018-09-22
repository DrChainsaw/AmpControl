package ampcontrol.model.training.listen;

import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import java.util.function.Consumer;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link ActivationContribution}
 *
 * @author Christian Skärby
 */
public class ActivationContributionTest {

    /**
     * Test that activation contribution is what is expected in a simple model with one dense layer where gradient and
     * activation is known beforehand. Use a real model as computations are trivial, but needs to be based on the right
     * input.
     */
    @Test
    public void dense() {
        final String inputName = "input";
        final String layerName = "layer";
        final String outputName = "output";
        final Layer layer = new DenseLayer.Builder().nIn(2)
                .nOut(2)
                .hasBias(false)
                .activation(new ActivationIdentity())
                .build();
        final ComputationGraph graph = getComputationGraph(inputName, layerName, outputName, InputType.feedForward(2), layer);

        graph.getLayer(layerName).setParam(DefaultParamInitializer.WEIGHT_KEY, Nd4j.eye(2));
        graph.getLayer(outputName).setParam(DefaultParamInitializer.WEIGHT_KEY, Nd4j.eye(2));

        final Probe probe = new Probe();
        final ActivationContribution activationContribution = new ActivationContribution(layerName, probe);
        graph.addListeners(activationContribution);
        final INDArray feature = Nd4j.ones(2, 2);
        final INDArray label = Nd4j.create(new double[][]{{1, -1}, {1, -1}});

        graph.fit(new DataSet(feature, label));

        activationContribution.onEpochEnd(graph);
        final INDArray expected = Nd4j.create(new double[]{0, 2}); // gradient is 0,2 and "activation" is 1,1
        probe.assertNrofCalls(1);
        assertEquals("Incorrect output!", expected, probe.last);
    }

    /**
     * Test a 2D convolution layer with bias. CBA to set this up so the output is known -> basically just a smoke test.
     */
    @Test
    public void convWithBias() {
        final String inputName = "input";
        final String layerName = "layer";
        final String outputName = "output";
        final int heigh = 23;
        final int width = 31;
        final int depth = 5;
        final int miniBatchSize = 3;
        final int nOut = 13;
        final Layer layer = new Convolution2D.Builder(7, 11)
                .nOut(nOut)
                .hasBias(true)
                .activation(new ActivationReLU())
                .build();
        final ComputationGraph graph = getComputationGraph(
                inputName,
                layerName,
                outputName,
                InputType.convolutional(heigh, width, depth), layer);

        final Probe probe = new Probe();
        final ActivationContribution activationContribution = new ActivationContribution(layerName, probe);
        graph.addListeners(activationContribution);
        final int nrofOutputs = graph.layerSize(outputName);
        final INDArray feature = Nd4j.linspace(-10, 10, heigh * width * depth * miniBatchSize).reshape(miniBatchSize, depth, heigh, width);
        final INDArray label = Nd4j.linspace(-2, 2, nrofOutputs * miniBatchSize).reshape(miniBatchSize, nrofOutputs);

        graph.fit(new DataSet(feature, label));

        activationContribution.onEpochEnd(graph);
        probe.assertNrofCalls(1);
        assertArrayEquals("Incorrect output shape", new long[] {1, nOut}, probe.last.shape());
    }

    @NotNull
    ComputationGraph getComputationGraph(
            String inputName,
            String layerName,
            String outputName,
            InputType inputType,
            Layer layer) {

        final String epsSpyName = "epsSpy";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(11))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .addLayer(layerName, layer, inputName)
                .addVertex(epsSpyName, new EpsilonSpyVertex(), layerName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .lossFunction(new LossMSE())
                        .hasBias(false)
                        .activation(new ActivationIdentity())
                        .build(), epsSpyName)
                .setInputTypes(inputType)
                .build());
        graph.init();
        return graph;
    }

    private static class Probe implements Consumer<INDArray> {

        private int nrofCalls;
        private INDArray last;

        @Override
        public void accept(INDArray indArray) {
            nrofCalls++;
            last = indArray;
        }

        private void assertNrofCalls(int expected) {
            assertEquals("Incorrect number of calls!", expected, nrofCalls);
        }
    }

}