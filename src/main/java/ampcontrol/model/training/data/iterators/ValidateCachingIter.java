package ampcontrol.model.training.data.iterators;

import ampcontrol.audio.processing.Pipe;
import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.Spectrogram;
import ampcontrol.audio.processing.UnitMaxZeroMean;
import ampcontrol.model.training.TrainingDescription;
import ampcontrol.model.training.data.*;
import ampcontrol.model.training.data.processing.SilenceProcessor;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;

/**
 * Test that {@link CachingDataSetIterator} does not impact the result of {@link Cnn2DDataSetIterator}. Should ideally
 * be a testcase, but requires a dataset to run.
 *
 * @author Christian Skärby
 */
public class ValidateCachingIter {

    public static void main(String[] args) throws InterruptedException {

        int clipLengthMs = 1000;
        int clipSamplingRate = 44100;
        Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
        //Path modelDir = Paths.get("E:\\Software projects\\java\\leadRythm\\RythmLeadSwitch\\models");
        List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");
        int trainingIterations = 20;
        int trainBatchSize = 32;
        int evalBatchSize = 1;
        double evalSetPercentage = 4;

        final int timeWindowSize = 100;

        final Supplier<ProcessingResult.Factory> audioPostProcSupplier = () -> new Pipe(
                new Spectrogram(512, 32),
                new UnitMaxZeroMean()
        );

        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, audioPostProcSupplier);
        Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        final int seed = 666;
        final DataProviderBuilder train1 = new TrainingDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, new SimpleStateFactory(seed));
        final DataProviderBuilder train2 = new TrainingDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, new SimpleStateFactory(seed));

        final DataProviderBuilder eval1 = new EvalDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, new SimpleStateFactory(seed));
        final DataProviderBuilder eval2 = new EvalDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, new SimpleStateFactory(seed));

        try {
            DataSetFileParser.parseFileProperties(baseDir, new TrainingDescription.DataSetMapper(train1, eval1, evalSetPercentage));
            DataSetFileParser.parseFileProperties(baseDir, new TrainingDescription.DataSetMapper(train2, eval2, evalSetPercentage));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Thread.sleep(5000);
        final int evalCacheSize = (int) (0.75 * (clipLengthMs / timeWindowSize * (eval1.getNrofFiles() / evalBatchSize)));
        final MiniEpochDataSetIterator evalIter = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(eval1.createProvider(), evalBatchSize, labels)
                ,evalCacheSize);

        evalIter.next();
        final MiniEpochDataSetIterator trainIterCache = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(
                        train1.createProvider(), trainBatchSize, labels),
                trainingIterations);

        final Cnn2DDataSetIterator trainIterNoCache = new Cnn2DDataSetIterator(
                train2.createProvider(), trainBatchSize, labels);
        int nrofExamplesToTest = 1000;
        for (int i = 0; i < nrofExamplesToTest; i++) {
            System.out.println("*************** start round " + i + " ****************");
            List<INDArray> featuresCache = new ArrayList<>();
            List<INDArray> featuresNoCache = new ArrayList<>();
            for (int j = 0; j < trainingIterations; j++) {
                DataSet cache = trainIterCache.next();
                System.out.println("Get non-cached!");
                DataSet noCache = trainIterNoCache.next();

                featuresCache.add(cache.getFeatures());
                featuresNoCache.add(noCache.getFeatures());
            }
            checkEquality(featuresCache, featuresNoCache);
            System.out.println("Example " + i + " ok!");
            trainIterCache.reset();

        }

    }


    private static void checkEquality(List<INDArray> arrs1, List<INDArray> arrs2) {
        for (INDArray arr1 : arrs1) {
            boolean match = false;
            for (INDArray arr2 : arrs2) {
                if (arr1.equalsWithEps(arr2, 1e-12)) {
                    System.out.println("Found match!");
                    match = true;
                    break;
                }
            }
            if (!match) {
                //System.out.println("Mistmatch! \n" + arr1.sum(0));
                throw new IllegalStateException("Mismatch!");
            }
        }
    }

}
