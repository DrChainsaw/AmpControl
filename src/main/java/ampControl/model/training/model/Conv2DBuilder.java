package ampControl.model.training.model;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationELU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/**
 * Builder for stacked 2D convolutions. Practically replaced everywhere by {@link BlockBuilder} but kept in the unlikely
 * event that someone finds this project and wants to use it but can't figure out the {@link BlockBuilder}. Trying to
 * add options for where to put batch normalization is what made me ditch this approach...
 *
 * @author Christian Skärby
 */
public class Conv2DBuilder {

    private final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();

    private double startingLearningRate = 0.05;

    public interface LearningRateSchedulePolicy {
        void apply(NeuralNetConfiguration.Builder aBuilder);
    }

    private LearningRateSchedulePolicy lrPolicy = new LearningRateSchedulePolicy() {
        private Map<Integer, Double> lrSchedule = new HashMap<>();

        {

            lrSchedule.put(0, startingLearningRate);
            lrSchedule.put(200, startingLearningRate/2);
            lrSchedule.put(1500, startingLearningRate/5);
            lrSchedule.put(10000, startingLearningRate/10);
            lrSchedule.put(100000, startingLearningRate/100);
            //lrSchedule.put(20000, startingLearningRate/1000);
            lrSchedule.put(500000, startingLearningRate/500);

        }

        @Override
        public void apply(Builder aBuilder) {
            aBuilder
                    .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                    .learningRateSchedule(lrSchedule);
        }
    };

    private String namePrefix = "";
    private int seed = 666;
    private int nrofIterations = 1;

    private int nrofCnnLayers = 4;

    private boolean useBatchNormalizationCnn = false;
    private boolean useBatchNormalizationDnn = false;

    private int nrofKernels = 64;
    private int nrofKernelsGrowth = 1;
    private int nrofConvLayersPerMaxPool = 1;
    private int kernelSize_h = 4;
    private int kernelSize_v = 4;

    private int maxPoolsSize_h = 2;
    private int maxPoolsSize_v = 2;
    private int[] inputShape;

    private int nrofDnnLayers = 1;
    private int dnnHiddenWidth = 256;

    private int nrofLabels;

    private IActivation convActivation = new ActivationELU();
    private IActivation dnnActivation = new ActivationReLU();
    private IUpdater updater = new Adam(startingLearningRate, 0.9, 0.999, 1e-8);

    private double accuracy = 0;
    private final Pattern accPattern = Pattern.compile("\\s*Accuracy:\\s*(\\d*,?\\d*)");

    public MultiLayerNetwork build(String modelDir) {
        File modelFile = new File(modelDir + File.separator + getName());
        if (modelFile.exists()) {
            try {
                System.out.println("restoring saved model: " + modelFile.getAbsolutePath());
                BufferedReader scoreReader = Files.newBufferedReader(Paths.get(modelFile.getAbsolutePath() + "_best.score"));

                accuracy = scoreReader.lines()
                        .map(line -> accPattern.matcher(line))
                        .filter(matcher -> matcher.matches())
                        .map(matcher -> matcher.group(1))
                        .map(str -> str.replace(",", "."))
                        .mapToDouble(str -> Double.parseDouble(str))
                        .findFirst()
                        .orElse(0);
                return ModelSerializer.restoreMultiLayerNetwork(modelFile, false);

            } catch (IOException e) {
                throw new RuntimeException("Failed to load model");
            }
        }
        System.out.println("Creating model: " + getName());

        builder.seed(seed)
                .iterations(nrofIterations)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY) // Will be set later on
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.regularization(true).l1(1e-5).l2(1e-5).dropOut(0.5)
                .updater(updater);

        builder.setTrainingWorkspaceMode(WorkspaceMode.SINGLE);
        builder.setInferenceWorkspaceMode(WorkspaceMode.SINGLE);
        lrPolicy.apply(builder);

        ListBuilder listBuilder = builder.list();

        addLayers(listBuilder);

        listBuilder
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]));

        MultiLayerNetwork model = new MultiLayerNetwork(listBuilder.build());
        model.init();
        return model;
    }

    private void addLayers(final ListBuilder listBuilder) {
        final int nrofCnnLayersPerInnerIter = useBatchNormalizationCnn ? 2: 1;
        final int nrofLayersPerMaxPool = nrofConvLayersPerMaxPool*nrofCnnLayersPerInnerIter;
        final int nrofCnnBlocksPerIter =  nrofLayersPerMaxPool + 1;

        IntStream.range(0, nrofCnnLayers)
                .map(layerInd -> nrofCnnBlocksPerIter * layerInd)
                .forEach(layerInd -> {
                    int cnnInd = layerInd / nrofCnnBlocksPerIter;

                    IntStream.range(0, nrofConvLayersPerMaxPool)
                            .map(lInd -> nrofCnnLayersPerInnerIter*lInd)
                            .map(lInd -> lInd + layerInd)
                            .forEach(cnnLayerInd -> {
                                int layerIndInner = cnnLayerInd;
                                System.out.println("cnn Layer: " + layerIndInner);
                                listBuilder
                                        .layer(layerIndInner++ , new ConvolutionLayer.Builder(kernelSize_h, kernelSize_v)
                                                .activation(convActivation)
                                                //.nIn(inputShape[2])
                                                .nOut(nrofKernels * (int) Math.pow(nrofKernelsGrowth, cnnInd))
                                                .build());
                                if (useBatchNormalizationCnn) {
                                    System.out.println("cnn BN Layer: " + layerIndInner );
                                    listBuilder
                                            .layer(layerIndInner++ , new BatchNormalization.Builder()
                                            .eps(1e-3)
                                            .build());
                                }
//                                System.out.println("cnn act Layer: " + layerIndInner);
//                                listBuilder
//                                        .layer(layerIndInner++ , new ActivationLayer.Builder()
//                                                .activation(convActivation)
//                                                .build());

                            });
                    final int mpLayerInd = layerInd + nrofCnnBlocksPerIter-1;
                    listBuilder
                            .layer(mpLayerInd, new SubsamplingLayer.Builder(PoolingType.MAX)
                                    .kernelSize(maxPoolsSize_h, maxPoolsSize_v)
                                    .stride(maxPoolsSize_h, maxPoolsSize_v)
                                    .build());
                    System.out.println("MP Layer: " + mpLayerInd);

                });

        final int nrofDnnLayersPerIter = useBatchNormalizationDnn ? 2 : 1;
        final int dnnStartInd = nrofCnnBlocksPerIter * nrofCnnLayers;
        IntStream.range(0, nrofDnnLayers)
                .map(layerInd -> nrofDnnLayersPerIter * layerInd + dnnStartInd)
                .forEach(layerInd -> {
                    int layerIndInner = layerInd;
                    System.out.println("dnn Layer: " + layerIndInner);
                    listBuilder
                            .layer(layerIndInner++, new DenseLayer.Builder()
                                    .activation(dnnActivation)
                                    .nOut(dnnHiddenWidth)
                                    .build());

                    if (useBatchNormalizationDnn) {
                        System.out.println("dnn BN Layer: " + layerIndInner);
                        listBuilder
                                .layer(layerIndInner++, new BatchNormalization());
                    }
//                    System.out.println("dnn act Layer: " + layerIndInner);
//                    listBuilder
//                            .layer(layerIndInner++, new ActivationLayer.Builder()
//                                    .activation(dnnActivation)
//                                    .build());

                });
        final int outputLayerInd = nrofDnnLayers * nrofDnnLayersPerIter + dnnStartInd;
        System.out.println("out Layer: " + outputLayerInd);
        listBuilder.layer(outputLayerInd,
                new OutputLayer.Builder()
                        .lossFunction(new LossMCXENT(Nd4j.create(new double[] {0.1,0.3, 0.6,1})))
                        .nOut(nrofLabels)
                        .activation(new ActivationSoftmax())
                        .build());


    }
//private final Pattern activationTypePatt = Pattern.compile(".*Activation(\\w*)");
    public String getName() {
        String kernelGrowth = nrofKernelsGrowth > 1 ? nrofKernelsGrowth + "kernelGrowth_" : "";
        String batchNormCnn = useBatchNormalizationCnn ? "cnnBN_" : "";
        String batchNormDnn = useBatchNormalizationDnn ? "dnnBN_" : "";
        String nrofCnnLayersPerMp = nrofConvLayersPerMaxPool > 1 ? nrofConvLayersPerMaxPool + "cnnPerMp_" : "";
        String cnnActivation = convActivation.getClass().getSimpleName().replace("Activation", "");
        return namePrefix + "Cnn2d_" +cnnActivation +"_" + nrofCnnLayers + "layers_" +nrofCnnLayersPerMp + batchNormCnn + kernelGrowth + nrofKernels + "kernels_" + kernelSize_h + "_" + kernelSize_v +
                "_mp" + maxPoolsSize_h + "_" + maxPoolsSize_v + "_dnn_" + nrofDnnLayers + "layers_" + batchNormDnn + dnnHiddenWidth + "dnnW";
    }

    public double getAccuracy() {
        return accuracy;
    }

    public Conv2DBuilder setNamePrefix(String prefix) {
        namePrefix = prefix;
        return this;
    }

    public Conv2DBuilder setStartingLearningRate(double startingLearningRate) {
        this.startingLearningRate = startingLearningRate;
        return this;
    }

    public Conv2DBuilder setLrPolicy(LearningRateSchedulePolicy lrPolicy) {
        this.lrPolicy = lrPolicy;
        return this;
    }

    public Conv2DBuilder setSeed(int seed) {
        this.seed = seed;
        return this;
    }

    public Conv2DBuilder setNrofIterations(int nrofIterations) {
        this.nrofIterations = nrofIterations;
        return this;
    }

    public Conv2DBuilder setUseBatchNormalizationCnn(boolean useBatchNormalizationCnn) {
        this.useBatchNormalizationCnn = useBatchNormalizationCnn;
        return this;
    }

    public Conv2DBuilder setUseBatchNormalizationDnn(boolean useBatchNormalizationDnn) {
        this.useBatchNormalizationDnn = useBatchNormalizationDnn;
        return this;
    }


    public Conv2DBuilder setNrofCnnLayers(int nrofLayers) {
        this.nrofCnnLayers = nrofLayers;
        return this;
    }

    public Conv2DBuilder setNrofKernels(int nrofKernels) {
        this.nrofKernels = nrofKernels;
        return this;
    }

    public Conv2DBuilder setNrofKernelsGrowth(int nrofKernelsGrowth) {
        this.nrofKernelsGrowth = nrofKernelsGrowth;
        return this;
    }

    public Conv2DBuilder setNrofConvLayersPerMaxPool(int nrofConvLayersPerMaxPool) {
        this.nrofConvLayersPerMaxPool = nrofConvLayersPerMaxPool;
        return this;
    }

    public Conv2DBuilder setKernelSize_h(int kernelSize_h) {
        this.kernelSize_h = kernelSize_h;
        return this;
    }

    public Conv2DBuilder setKernelSize_v(int kernelSize_v) {
        this.kernelSize_v = kernelSize_v;
        return this;
    }

    public Conv2DBuilder setMaxPoolsSize_h(int maxPoolsSize_h) {
        this.maxPoolsSize_h = maxPoolsSize_h;
        return this;
    }

    public Conv2DBuilder setMaxPoolsSize_v(int maxPoolsSize_v) {
        this.maxPoolsSize_v = maxPoolsSize_v;
        return this;
    }

    public Conv2DBuilder setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
        return this;
    }

    public Conv2DBuilder setConvActivation(IActivation convActivation) {
        this.convActivation = convActivation;
        return this;
    }

    public Conv2DBuilder setUpdater(IUpdater updater) {
        this.updater = updater;
        return this;
    }

    public Conv2DBuilder setNrofDnnLayers(int nrofDnnLayers) {
        this.nrofDnnLayers = nrofDnnLayers;
        return this;
    }

    public Conv2DBuilder setDnnHiddenWidth(int dnnHiddenWidth) {
        this.dnnHiddenWidth = dnnHiddenWidth;
        return this;
    }

    public Conv2DBuilder setNrofLabels(int nrofLabels) {
        this.nrofLabels = nrofLabels;
        return this;
    }

    public Conv2DBuilder setDnnActivation(IActivation dnnActivation) {
        this.dnnActivation = dnnActivation;
        return this;
    }

}
