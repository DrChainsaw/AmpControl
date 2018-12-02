package ampcontrol.model.training.model.evolve;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

    /**
     * Create a model with residual convolution layers
     *
     * @param name1 Name of first layer
     * @param name2 Name of second layer
     * @param name3 Name of third layer
     * @return a {@link ComputationGraph}
     */
    public static ComputationGraph getResNet(String name1, String name2, String name3) {
        final String convIn = "convIn";
        final String rbAdd0 = "rbAdd0";
        final String rbAdd1 = "rbAdd1";
        final String rbAdd2 = "rbAdd2";
        final int resNout = 7;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .biasInit(666)
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

    /**
     * Returns a {@link ComputationGraph} with convolution layers in a forked connection
     *
     * @param beforeFork Name of layer before fork
     * @param afterFork  Name of layer after fork
     * @param forkNames  Names of layers in fork (one path per layer)
     * @return A {@link ComputationGraph}
     */
    public static ComputationGraph getForkNet(String beforeFork, String afterFork, String... forkNames) {
        final int forkNoutStart = 7;
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .biasInit(666)
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(beforeFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(4)
                        .build(), inputName)
                .addLayer(batchNormName, new BatchNormalization.Builder().build(), beforeFork);

        final String[] forkBatchNormNames = Stream.of(forkNames).map(name -> name + batchNormName)
                .collect(Collectors.toList()).toArray(new String[forkNames.length]);
        for (int i = 0; i < forkNames.length; i++) {
            builder.addLayer(forkNames[i], new Convolution2D.Builder(forkNames.length - i, 1 + i)
                    .convolutionMode(ConvolutionMode.Same)
                    .nOut(forkNoutStart + i)
                    .build(), batchNormName);

            builder.addLayer(
                    forkBatchNormNames[i], new BatchNormalization.Builder().build(), forkNames[i]);
        }

        final ComputationGraph graph = new ComputationGraph(builder
                .addLayer(afterFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(3)
                        .build(), forkBatchNormNames)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), afterFork)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(9)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), denseName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(beforeFork).getParam(W), true);
        setToLinspace(graph.getLayer(beforeFork).getParam(B), false);
        setToLinspace(graph.getLayer(afterFork).getParam(W), false);
        setToLinspace(graph.getLayer(afterFork).getParam(B), true);
        for (int i = 0; i < forkNames.length; i++) {
            setToLinspace(graph.getLayer(forkNames[i]).getParam(W), i % 2 == 0);
            setToLinspace(graph.getLayer(forkNames[i]).getParam(B), (i + 1) % 2 == 0);
        }
        return graph;
    }

    /**
     * Returns a {@link ComputationGraph} with a residual block consisting of convolution layers in a forked connection
     *
     * @param beforeFork Name of layer before fork
     * @param afterFork  Name of layer after fork
     * @param forkNames  Names of layers in fork (one path per layer)
     * @return A {@link ComputationGraph}
     */
    public static ComputationGraph getForkResNet(String beforeFork, String afterFork, String... forkNames) {
        final int[] forkNouts = IntStream.range(0, forkNames.length).map(i -> i + 5).toArray();
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .biasInit(666)
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(beforeFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(IntStream.of(forkNouts).sum())
                        .build(), inputName);

        for (int i = 0; i < forkNames.length; i++) {
            final String bnName = forkNames[i] + batchNormName;
            builder
                    .addLayer(bnName, new BatchNormalization.Builder().build(), beforeFork)
                    .addLayer(forkNames[i], new Convolution2D.Builder(forkNames.length - i, 1 + i)
                            .convolutionMode(ConvolutionMode.Same)
                            .nOut(forkNouts[i])
                            .build(), bnName);

        }

        final String mergeName = "mergeFork";
        builder.addVertex(mergeName, new MergeVertex(), forkNames);

        final String elemAddName = "addMergeAndBeforeFork";
        builder.addVertex(elemAddName, new ElementWiseVertex(ElementWiseVertex.Op.Add), mergeName, beforeFork);

        final ComputationGraph graph = new ComputationGraph(builder
                .addLayer(afterFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(3)
                        .build(), elemAddName)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), afterFork)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(9)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), denseName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(beforeFork).getParam(W), true);
        setToLinspace(graph.getLayer(beforeFork).getParam(B), false);
        setToLinspace(graph.getLayer(afterFork).getParam(W), false);
        setToLinspace(graph.getLayer(afterFork).getParam(B), true);
        for (int i = 0; i < forkNames.length; i++) {
            setToLinspace(graph.getLayer(forkNames[i]).getParam(W), i % 2 == 0);
            setToLinspace(graph.getLayer(forkNames[i]).getParam(B), (i + 1) % 2 == 0);
        }
        return graph;
    }

    /**
     * Returns a {@link ComputationGraph} with a residual block consisting of convolution layers in a forked connection
     *
     * @param beforeFork Name of layer before fork
     * @param afterFork  Name of layer after fork
     * @param forkNames  Names of layers in fork (one path per layer)
     * @return A {@link ComputationGraph}
     */
    public static ComputationGraph getDoubleForkResNet(String beforeFork, String afterFork, String... forkNames) {
        final int[] forkNouts = IntStream.range(0, forkNames.length).map(i -> i + 5).toArray();
        final int forkSize = IntStream.of(forkNouts).sum();
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .biasInit(666)
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(beforeFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(forkSize)
                        .build(), inputName);
               // .addLayer(batchNormName, new BatchNormalization(), beforeFork);

        for (int i = 0; i < forkNames.length; i++) {
            final String bnName = beforeFork;//forkNames[i] + batchNormName;
            builder
                    //.addLayer(bnName, new BatchNormalization.Builder().build(), batchNormName)
                    .addLayer(forkNames[i], new Convolution2D.Builder(forkNames.length - i, 1 + i)
                            .convolutionMode(ConvolutionMode.Same)
                            .nOut(forkNouts[i])
                            .build(), bnName);

        }
        final String[] bnNames = new String[(int)Math.ceil(forkNames.length/2d)];
        for (int i = 0; i < bnNames.length; i++) {
            final int ind = i;
            final String mergeNameInner = "mergeInner" + ind;
            bnNames[i] = mergeNameInner;//batchNormName + forkNames[ind];
            final String[] toMerge = IntStream.range(0, forkNames.length)
                    .filter(j -> j / 2 == ind)
                    .mapToObj(j -> forkNames[j])
                    .toArray(String[]::new);
            builder.addVertex(mergeNameInner, new MergeVertex(),  toMerge);
                 //   .addLayer(bnNames[i], new BatchNormalization.Builder().build(), mergeNameInner);
        }

        final String mergeName = "mergeFork";
        builder.addVertex(mergeName, new MergeVertex(), bnNames);

        final String elemAddName = "addMergeAndBeforeFork";
        builder.addVertex(elemAddName, new ElementWiseVertex(ElementWiseVertex.Op.Add), mergeName,beforeFork); //batchNormName);

        final int nextForkNrofPaths = 4;
        final String[] nextFork = new String[nextForkNrofPaths];
        final int nextForkAvgNout = forkSize / nextForkNrofPaths;
        int reminder = forkSize;
        for (int i = 0; i < nextForkNrofPaths; i++) {
            final int size = i == nextForkNrofPaths-1 ? reminder : nextForkAvgNout;
            reminder -= size;
            nextFork[i]  = "nextFork" + i;
            builder.addLayer(nextFork[i],  new Convolution2D.Builder(1,1)
                    .convolutionMode(ConvolutionMode.Same)
                    .nOut(size)
                    .build(), elemAddName);
        }

        final String nextMergeName = "mergeNextFork";
        builder.addVertex(nextMergeName, new MergeVertex(), nextFork);

        final String nextElemAddName = "addNextMergeAndBeforeFork";
        builder.addVertex(nextElemAddName, new ElementWiseVertex(ElementWiseVertex.Op.Add), nextMergeName, beforeFork);//batchNormName);

        final ComputationGraph graph = new ComputationGraph(builder
                .addLayer(afterFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(3)
                        .build(), nextElemAddName)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), afterFork)
                .addLayer(denseName, new DenseLayer.Builder()
                        .nOut(9)
                        .build(), globPoolName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), denseName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(beforeFork).getParam(W), true);
        setToLinspace(graph.getLayer(beforeFork).getParam(B), false);
        setToLinspace(graph.getLayer(afterFork).getParam(W), false);
        setToLinspace(graph.getLayer(afterFork).getParam(B), true);
        for (int i = 0; i < forkNames.length; i++) {
            setToLinspace(graph.getLayer(forkNames[i]).getParam(W), i % 2 == 0);
            setToLinspace(graph.getLayer(forkNames[i]).getParam(B), (i + 1) % 2 == 0);
        }
        graph.outputSingle(Nd4j.randn(new long[] {1, 3, 33, 33}));
        return graph;
    }

    /**
     * Returns a {@link ComputationGraph} with a complex structure for stress testing
     *
     * @param beforeFork Name of layer before first fork
     * @param afterFork  Name of layer after last fork
     * @param fork1Names Names of layers in first fork (one path per layer)
     * @param fork2Names Names of layers in first fork (one path per layer)
     * @return A {@link ComputationGraph}
     */
    public static ComputationGraph getForkResOuterInnerNet(String beforeFork, String afterFork, String[] fork1Names, String[] fork2Names) {

        final int[] fork1Nouts = IntStream.range(0, fork1Names.length).map(i -> 7 + i).toArray();
        final int[] fork2Nouts = IntStream.range(0, fork2Names.length).map(i -> 3 + i).toArray();
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .biasInit(666)
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(beforeFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(IntStream.of(fork1Nouts).sum())
                        .build(), inputName)
                .addLayer(batchNormName, new BatchNormalization.Builder().build(), beforeFork);

        for (int i = 0; i < fork1Names.length; i++) {
            builder.addLayer(fork1Names[i], new Convolution2D.Builder(2, 2)
                    .convolutionMode(ConvolutionMode.Same)
                    .nOut(fork1Nouts[i])
                    .build(), batchNormName);
        }

        final String fork1MergeName = "fork1Merge";
        builder.addVertex(fork1MergeName, new MergeVertex(), fork1Names);

        final String elemAdd1Name = "elemAdd1";
        builder.addVertex(elemAdd1Name, new ElementWiseVertex(ElementWiseVertex.Op.Add), fork1MergeName, batchNormName);

        // Not handled: Leads to double change in afterFork2 if anything which touches elemAdd1 is changed
        //final String fork1AndInputMergeName = "fork1AndInputMerge";
        //builder.addVertex(fork1AndInputMergeName, new MergeVertex(), fork1MergeName, batchNormName);

        final String[] fork2ElemAdds = Stream.of(fork2Names).map(name -> name + "Add")
                .collect(Collectors.toList()).toArray(new String[fork2Names.length]);
        for (int i = 0; i < fork2Names.length; i++) {

            final String fork2ResName = "res" + fork2Names[i];
            builder.addLayer(fork2Names[i], new Convolution2D.Builder(1, 1)
                    .convolutionMode(ConvolutionMode.Same)
                    .nOut(fork2Nouts[i])
                    .build(), fork1MergeName)
                    .addLayer(fork2ResName, new Convolution2D.Builder(3, 3)
                    .convolutionMode(ConvolutionMode.Same)
                    .nOut(fork2Nouts[i])
                    .build(), fork2Names[i])
                    .addVertex(fork2ElemAdds[i], new ElementWiseVertex(ElementWiseVertex.Op.Add), fork2Names[i], fork2ResName);
        }
        final String[] afterFork2 = Stream.concat(Stream.of(elemAdd1Name), Stream.of(fork2ElemAdds))
                .collect(Collectors.toList()).toArray(new String[fork2Names.length + 1]);

        final ComputationGraph graph = new ComputationGraph(builder
                .addLayer(afterFork, new Convolution2D.Builder(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(3)
                        .build(), afterFork2)
                .addLayer(globPoolName, new GlobalPoolingLayer.Builder().build(), afterFork)
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
