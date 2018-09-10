package ampcontrol.model.training.model.evolve;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Utils for doing testing on Compu
 */
public class GraphUtils {
    private final static String inputName = "input";
    private final static String outputName = "output";
    private final static String poolName = "pool";
    private final static String batchNormName = "batchNorm";
    private final static String globPoolName = "globPool";
    private final static String denseName = "dense";
    public final static String W = DefaultParamInitializer.WEIGHT_KEY;
    public final static String B = DefaultParamInitializer.BIAS_KEY;

    /**
     * Returns a CNN graph with pooling and batchnorm layers
     *
     * @param cnn1Name Name of first conv layer
     * @param cnn2Name Name of first conv layer
     * @param cnn3Name Name of first conv layer
     * @return a CNN graph
     */
    @NotNull
    public static ComputationGraph getCnnGraph(String cnn1Name, String cnn2Name, String cnn3Name) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(cnn1Name, new Convolution2D.Builder(3, 3)
                        .nOut(10)
                        .build(), inputName)
                .addLayer(batchNormName, new BatchNormalization.Builder().build(), cnn1Name)
                .addLayer(cnn2Name, new Convolution2D.Builder(1, 1)
                        .nOut(5)
                        .build(), batchNormName)
                .addLayer(poolName, new SubsamplingLayer.Builder().build(), cnn2Name)
                .addLayer(cnn3Name, new Convolution2D.Builder(2, 2)
                        .nOut(7)
                        .build(), poolName)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), cnn3Name)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(9)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), denseName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(cnn1Name).getParam(W), true);
        setToLinspace(graph.getLayer(cnn2Name).getParam(W), false);
        setToLinspace(graph.getLayer(cnn3Name).getParam(W), true);
        setToLinspace(graph.getLayer(cnn1Name).getParam(B), false);
        setToLinspace(graph.getLayer(cnn2Name).getParam(B), true);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.BETA).addi(2);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.GAMMA).addi(5);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.GLOBAL_MEAN).addi(7);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.GLOBAL_VAR).muli(9);
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
