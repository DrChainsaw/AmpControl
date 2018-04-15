package ampControl.model.training;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.Random;
import java.util.function.Supplier;

import ampControl.audio.processing.*;
import ampControl.model.training.data.*;
import ampControl.model.training.model.*;
import ampControl.model.training.model.description.*;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

import ampControl.model.training.data.AudioDataProvider.AudioProcessorBuilder;
import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.data.iterators.Cnn2DDataSetIterator;
import ampControl.model.training.data.processing.SilenceProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main "app" for training. Describes what models to use and how they shall be trained.
 * TODO: Make parameters setable through Jcommander?
 *
 * @author Christian Skärby
 */
public class TrainingDescription {

    private static final Logger log = LoggerFactory.getLogger(TrainingDescription.class);

    private final static int clipLengthMs = 1000;
    private final static int clipSamplingRate = 44100;
    private final static Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
    private final static Path modelDir = Paths.get("E:\\Software projects\\java\\leadRythm\\RythmLeadSwitch\\models");
    private final static List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");
    private final static int trainingIterations = 10; // 10
    private final static int trainBatchSize = 20;   // 32 64
    private final static int evalBatchSize = 20;
    private final static double evalSetPercentage = 3;

    /**
     * Maps a double valued identifier to a training or evaluation set respectively.
     *
     * @author Christian Skärby
     */
    public final static class DataSetMapper implements Function<Double, DataProviderBuilder> {

        private final DataProviderBuilder train;
        private final DataProviderBuilder eval;
        private final double evalPercentage;

        public DataSetMapper(DataProviderBuilder train, DataProviderBuilder eval, double evalPercentage) {
            this.train = train;
            this.eval = eval;
            this.evalPercentage = evalPercentage;
        }

        @Override
        public DataProviderBuilder apply(Double setId) {
            if (setId < evalPercentage) {
                return eval;
            }
            return train;
        }
    }

    public static void main(String[] args) {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
        CudaEnvironment.getInstance().getConfiguration()
                .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
                .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
                .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
                .setMaximumHostCache(6L * 1024 * 1024 * 1024L);

        List<ModelHandle> modelData = new ArrayList<>();

        final int trainingSeed = new Random().nextInt();
        //final int trainingSeed = 666;
        final int timeWindowSizeMs = 50;


//        final Supplier<ProcessingResult.Processing> audioPostProcessingSupplier = () -> new Pipe(
//                //      new Pipe(
//                new Spectrogram(256, 32),
//                //new Mfsc(clipSamplingRate)),
//                new LogScale()
//                //  ),
//
//                //  new UnitStdZeroMean()
//        );

        final Supplier<ProcessingResult.Processing> audioPostProcessingSupplier = () -> new Pipe(
                new Spectrogram(256, 16),
                new Fork(
                        new Pipe(
                                new Log10(),
                                new ZeroMean()),
                        new Pipe(
                                new Mfsc(clipSamplingRate),
                                new ZeroMean()
                        )
                )
        );

        //  final Supplier<ProcessingResult.Processing> audioPostProcessingSupplier = () -> new UnitMaxZeroMean();


//                () -> new Pipe(
//                new Spectrogrammm(512, 32, 1),
//                new UnitStdZeroMean()
//        );

        createModels(audioPostProcessingSupplier, timeWindowSizeMs, modelData, trainingSeed);


        //NativeOpsHolder.getInstance().getDeviceNativeOps().setOmpNumThreads(1);

        for (ModelHandle md : modelData) {
            log.info(md.name() + ": score: " + md.getBestEvalScore());
        }

        TrainingHarness harness = new TrainingHarness(modelData, modelDir.toAbsolutePath().toString());
        harness.startTraining();
    }

    private static void createModels(final Supplier<ProcessingResult.Processing> audioPostProcessingSupplier, final int timeWindowSize, List<ModelHandle> modelData, int trainingSeed) {
        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, audioPostProcessingSupplier);
        Map<String, AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        MultiplyLabelExpander labelExpander = new MultiplyLabelExpander()
                .addExpansion("rythm", 2)
                .addExpansion("lead", 2);
        MultiplyLabelExpander labelExpanderEval = new MultiplyLabelExpander()
                .addExpansion("noise", 20)
                .addExpansion("rythm", 100)
                .addExpansion("lead", 100);

        final Random volScaleRng = new Random(trainingSeed + 1);
        final Supplier<ProcessingResult.Processing> trainSupplier = () -> new Pipe(
                new RandScale(1000, 10, volScaleRng),
                audioPostProcessingSupplier.get()
        );

        final DataProviderBuilder train = new TrainingDataProviderBuilder(labelToBuilder, labelExpander, clipLengthMs, timeWindowSize, trainSupplier, trainingSeed);
        final DataProviderBuilder eval = new EvalDataProviderBuilder(labelToBuilder, labelExpanderEval, clipLengthMs, timeWindowSize, audioPostProcessingSupplier, 666);

        try {
            DataSetFileParser.parseFileProperties(baseDir, new DataSetMapper(train, eval, evalSetPercentage));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        final CachingDataSetIterator trainIter = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(
                        train.createProvider(), trainBatchSize, labels),
                trainingIterations);

        final int evalCacheSize = (int) (0.75 * (clipLengthMs / timeWindowSize * (eval.getNrofFiles() / evalBatchSize)));
        final CachingDataSetIterator evalIter = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(eval.createProvider(), evalBatchSize, labels),
                evalCacheSize);

        log.info("Nrof eval files: " + eval.getNrofFiles());

        // This knowledge needs to move somewhere else when multiple inputs are implemented
        final double[][] inputProto = silence.getResult().get().get(0);
        final int[] inputShape = {inputProto.length, inputProto[0].length, silence.getResult().get().size()};

        String prefix = "ws_" + timeWindowSize + SupplierFactory.prefix() + audioPostProcessingSupplier.get().name() + "_";

        //new ResNetConv2DFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        //new StackedConv2DFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new Conv1DLstmDenseFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

         new DenseNetFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

        // new Conv2DShallowWideFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new SoundnetFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

        // new SampleCnnFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

        // new SampleCnn2DFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new LstmTimeSeqFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
    }

}
