package ampcontrol.model.training.model.evolve;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Utils for doing testing on Compu
 *
 */
public class GraphUtils {
    private final static String inputName = "input";
    private final static String outputName = "output";
    private final static String poolName = "pool";
    public final static String W = DefaultParamInitializer.WEIGHT_KEY;
    public final static String B = DefaultParamInitializer.BIAS_KEY;

    @NotNull
    public static ComputationGraph getCnnGraph(String mutationName, String nextMutationName, String afterName) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(mutationName, new Convolution2D.Builder(3, 3)
                        .nOut(10)
                        .build(), inputName)
                .addLayer(nextMutationName, new Convolution2D.Builder(1, 1)
                        .nOut(5)
                        .build(), mutationName)
                .addLayer(poolName, new SubsamplingLayer.Builder().build(), nextMutationName)
                .addLayer(afterName, new Convolution2D.Builder(2, 2)
                        .nOut(7)
                        .build(), poolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), afterName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(mutationName).getParam(W), true);
        setToLinspace(graph.getLayer(nextMutationName).getParam(W), false);
        setToLinspace(graph.getLayer(afterName).getParam(W), true);
        setToLinspace(graph.getLayer(mutationName).getParam(B), false);
        setToLinspace(graph.getLayer(nextMutationName).getParam(B), true);
        return graph;
    }

    private static void setToLinspace(INDArray array, boolean reverse) {
        final long nrofElemsSource = Arrays.stream(array.shape()).reduce((i1, i2) -> i1 * i2).orElseThrow(() -> new IllegalArgumentException("No elements!"));
        array.assign(Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(array.shape()));
        if (reverse) {
            Nd4j.reverse(array);
        }
    }
}
