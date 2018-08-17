package ampcontrol.model.training;

import ampcontrol.audio.processing.*;
import ampcontrol.model.training.data.AudioDataProvider.AudioProcessorBuilder;
import ampcontrol.model.training.data.*;
import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.Cnn2DDataSetIterator;
import ampcontrol.model.training.data.processing.SilenceProcessor;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.description.ResNetConv2DFactory;
import ampcontrol.model.training.model.validation.listen.BufferedTextWriter;
import ampcontrol.model.visualize.RealTimePlot;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;

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
    private final static int trainBatchSize = 64;   // 32 64
    private final static int evalBatchSize = 64;
    private final static double evalSetPercentage = 4;

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

        List<ModelHandle> modelData = new ArrayList<>();

        final int trainingSeed = new Random().nextInt();
        //final int trainingSeed = 666;
        final int timeWindowSizeMs = 50;

        final ProcessingResult.Factory audioPostProcessingFactory = getAudioProcessingFactory();

        createModels(audioPostProcessingFactory, timeWindowSizeMs, modelData, trainingSeed);

        TrainingHarness harness = new TrainingHarness(modelData,
                modelDir.toAbsolutePath().toString(),
                title -> new RealTimePlot<>(title, modelDir.toAbsolutePath().toString() + File.separator + "plots"),
                BufferedTextWriter.defaultFactory);
        harness.startTraining(100000);
    }

    @NotNull
    private static ProcessingResult.Factory getAudioProcessingFactory() {
        return new Pipe(
                new Spectrogram(256, 16),
                new Fork(
                        new Fork(
                                new Pipe(
                                        new Log10(),
                                        new ZeroMean()),
                                new Pipe(
                                        new Mfsc(clipSamplingRate),
                                        new ZeroMean())
                        ),
                        new Pipe(
                                new Ycoord(),
                                new UnitMaxZeroMean()
                        )
                )
        );
    }

    private static void createModels(final ProcessingResult.Factory audioPostProcessingFactory, final int timeWindowSize, List<ModelHandle> modelData, int trainingSeed) {
        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, () -> audioPostProcessingFactory);
        Map<String, AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        MultiplyLabelExpander labelExpander = new MultiplyLabelExpander()
                .addExpansion("noise", 20)
                .addExpansion("rythm", 100)
                .addExpansion("lead", 100);
        MultiplyLabelExpander labelExpanderEval = new MultiplyLabelExpander()
                .addExpansion("noise", 20)
                .addExpansion("rythm", 100)
                .addExpansion("lead", 100);

        final Random volScaleRng = new Random(trainingSeed + 1);
        final ProcessingResult.Factory trainFactory = new Pipe(
                new RandScale(1000, 10, volScaleRng),
                audioPostProcessingFactory
        );

        final DataProviderBuilder train = new TrainingDataProviderBuilder(labelToBuilder, labelExpander, clipLengthMs, timeWindowSize, () -> trainFactory, trainingSeed);
        final DataProviderBuilder eval = new EvalDataProviderBuilder(labelToBuilder, labelExpanderEval, clipLengthMs, timeWindowSize, () -> audioPostProcessingFactory, 666);

        try {
            DataSetFileParser.parseFileProperties(baseDir, new DataSetMapper(train, eval, evalSetPercentage));
        } catch (IOException e) {
            throw new IllegalArgumentException(e);
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
        final double[][] inputProto = silence.getResult().stream().findFirst().orElseThrow(() -> new IllegalStateException("No input!"));
        final int[] inputShape = {inputProto.length, inputProto[0].length, (int) silence.getResult().stream().count()};

        String prefix = "ws_" + timeWindowSize + ProcessingFactoryFromString.prefix() + audioPostProcessingFactory.name() + "_";

       // new StackedConv2DFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        new ResNetConv2DFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        //new InceptionResNetFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new Conv1DLstmDenseFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new DenseNetFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

        // new Conv2DShallowWideFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new SoundnetFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

        // new SampleCnnFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);

        // new SampleCnn2DFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
        // new LstmTimeSeqFactory(trainIter, evalIter, inputShape, prefix, modelDir).addModelData(modelData);
    }

}
