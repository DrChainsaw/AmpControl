package ampcontrol.model.training.model.evolve;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
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
     * @param conv1Name Name of first conv layer
     * @param conv2Name Name of second conv layer
     * @param conv3Name Name of third conv layer
     * @return a CNN graph
     */
    @NotNull
    public static ComputationGraph getCnnGraph(String conv1Name, String conv2Name, String conv3Name) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(conv1Name, new Convolution2D.Builder(3, 3)
                        .nOut(10)
                        .build(), inputName)
                .addLayer(batchNormName, new BatchNormalization.Builder().build(), conv1Name)
                .addLayer(conv2Name, new Convolution2D.Builder(1, 1)
                        .nOut(5)
                        .build(), batchNormName)
                .addLayer(poolName, new SubsamplingLayer.Builder().build(), conv2Name)
                .addLayer(conv3Name, new Convolution2D.Builder(2, 2)
                        .nOut(7)
                        .build(), poolName)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), conv3Name)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(9)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), denseName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(conv1Name).getParam(W), true);
        setToLinspace(graph.getLayer(conv2Name).getParam(W), false);
        setToLinspace(graph.getLayer(conv3Name).getParam(W), true);
        setToLinspace(graph.getLayer(conv1Name).getParam(B), false);
        setToLinspace(graph.getLayer(conv2Name).getParam(B), true);
        setToLinspace(graph.getLayer(conv3Name).getParam(B), false);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.BETA).addi(2);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.GAMMA).addi(5);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.GLOBAL_MEAN).addi(7);
        graph.getLayer(batchNormName).getParam(BatchNormalizationParamInitializer.GLOBAL_VAR).muli(9);
        return graph;
    }


    /**
     * Returns a graph with only dense layers
     *
     * @param name1 name of first dense layer
     * @param name2 name of second dense layer
     * @param name3 name of third dense layer
     * @return a graph
     */
    @NotNull
    public static ComputationGraph getGraph(String name1, String name2, String name3) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.feedForward(33))
                .addLayer(name1, new DenseLayer.Builder()
                        .nOut(10)
                        .build(), inputName)
                .addLayer(name2, new DenseLayer.Builder()
                        .nOut(5)
                        .build(), name1)
                .addLayer(name3, new DenseLayer.Builder()
                        .nOut(7)
                        .build(), name2)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), name3)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(name1).getParam(W), true);
        setToLinspace(graph.getLayer(name1).getParam(B), false);
        setToLinspace(graph.getLayer(name2).getParam(W), false);
        setToLinspace(graph.getLayer(name2).getParam(B), true);
        return graph;
    }

    /**
     * Returns a graph with only dense layers where the last named layer is a {@link CenterLossOutputLayer}.
     *
     * @param name1 name of first dense layer
     * @param name2 name of second dense layer
     * @param name3 name of the output layer
     * @return a graph
     */
    @NotNull
    public static ComputationGraph getGraphNearOut(String name1, String name2, String name3) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(name3)
                .setInputTypes(InputType.feedForward(33))
                .addLayer(name1, new DenseLayer.Builder()
                        .nOut(10)
                        .build(), inputName)
                .addLayer(name2, new DenseLayer.Builder()
                        .nOut(5)
                        .build(), name1)
                .addLayer(name3, new CenterLossOutputLayer.Builder()
                        .nOut(7)
                        .build(), name2)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(name1).getParam(W), true);
        setToLinspace(graph.getLayer(name1).getParam(B), false);
        setToLinspace(graph.getLayer(name2).getParam(W), false);
        setToLinspace(graph.getLayer(name2).getParam(B), true);
        return graph;
    }

    /**
     * Returns a graph with only dense layers
     *
     * @param name1 name of first (conv) layer
     * @param name2 name of second (dense) layer
     * @param name3 name of third (dense) layer
     * @return a graph
     */
    @NotNull
    public static ComputationGraph getConvToDenseGraph(String name1, String name2, String name3) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(9, 9, 3))
                .addLayer(name1, new ConvolutionLayer.Builder(3, 3)
                        .nOut(10)
                        .build(), inputName)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), name1)
                .addLayer(name2, new DenseLayer.Builder()
                        .nOut(5)
                        .build(), globPoolName)
                .addLayer(name3, new DenseLayer.Builder()
                        .nOut(7)
                        .build(), name2)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), name3)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(name1).getParam(W), true);
        setToLinspace(graph.getLayer(name1).getParam(B), false);
        setToLinspace(graph.getLayer(name2).getParam(W), false);
        setToLinspace(graph.getLayer(name2).getParam(B), true);
        return graph;
    }

    public static ComputationGraph getResNet(String name1, String name2, String name3) {
        final String convIn = "convIn";
        final String rbAdd0 = "rbAdd0";
        final String rbAdd1 = "rbAdd1";
        final String rbAdd2 = "rbAdd2";
        final int resNout = 7;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(convIn, new Convolution2D.Builder(1, 1)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(resNout)
                        .build(), inputName)
                .addLayer(name1, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(resNout)
                        .build(), convIn)
                .addLayer(batchNormName, new BatchNormalization.Builder().build(), name1)
                .addVertex(rbAdd0, new ElementWiseVertex(ElementWiseVertex.Op.Add), batchNormName, convIn)
                .addLayer(name2, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(resNout)
                        .build(), rbAdd0)
                .addVertex(rbAdd1, new ElementWiseVertex(ElementWiseVertex.Op.Add), rbAdd0, name2)
                .addLayer(poolName, new SubsamplingLayer.Builder().build(), rbAdd1)
                .addLayer(name3, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(resNout)
                        .build(), poolName)
                .addVertex(rbAdd2, new ElementWiseVertex(ElementWiseVertex.Op.Add), poolName, name3)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), rbAdd2)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(9)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), denseName)
                .build());
        graph.init();
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
