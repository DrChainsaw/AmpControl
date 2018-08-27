package ampcontrol.model.training.data.iterators.validate;

import ampcontrol.audio.processing.Pipe;
import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.Spectrogram;
import ampcontrol.audio.processing.UnitMaxZeroMean;
import ampcontrol.model.training.data.AudioDataProvider;
import ampcontrol.model.training.data.DataProviderBuilder;
import ampcontrol.model.training.data.DataSetFileParser;
import ampcontrol.model.training.data.TrainingDataProviderBuilder;
import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.Cnn2DDataSetIterator;
import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.processing.SilenceProcessor;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import org.apache.commons.lang.mutable.MutableInt;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertTrue;

/**
 * Test that {@link CachingDataSetIterator} does not impact the result of {@link Cnn2DDataSetIterator}. Should ideally
 * be a testcase, but requires a dataset to run.
 *
 * @author Christian Skärby
 */
public class ValidateCachingIter {

    private static final Logger log = LoggerFactory.getLogger(ValidateAsynchIter.class);

    public static void main(String[] args) {

        int clipLengthMs = 1000;
        int clipSamplingRate = 44100;
        Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
        List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");
        int trainingIterations = 20;
        int trainBatchSize = 64;

        final int timeWindowSize = 100;

        final Supplier<ProcessingResult.Factory> audioPostProcSupplier = () -> new Pipe(
                new Spectrogram(8, 512),
                new UnitMaxZeroMean());

        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, audioPostProcSupplier);
        Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        final int seed = 666;
        final DataProviderBuilder train1 = new TrainingDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, new SimpleStateFactory(seed));
        final DataProviderBuilder train2 = new TrainingDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, new SimpleStateFactory(seed));

        mapFiles(train1, baseDir);
        mapFiles(train2, baseDir);

        final MiniEpochDataSetIterator iterCache = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(
                        train1.createProvider(), trainBatchSize, labels),
                trainingIterations);

        final Cnn2DDataSetIterator iterNoCache = new Cnn2DDataSetIterator(
                train2.createProvider(), trainBatchSize, labels);


        final int nrofExamplesToTest = 1000;
        performValidation(iterCache, iterNoCache, nrofExamplesToTest, ValidateAsynchIter::checkSumEquality);
    }

    /**
     * Test that a {@link CachingDataSetIterator} gives same feature->label pairs as its source iterator. Basically a
     * unit test for some of the functionality of this class
     */
    @Test
    public void resetAndResetMiniEpoch() {
        final MutableInt stateCache = new MutableInt(0);
        final MutableInt stateNoCache = new MutableInt(0);
        performValidation(
                new CachingDataSetIterator(createTestDataSetIterator(() -> stateCache), 3),
                createTestDataSetIterator(() -> stateNoCache),
                10,
                ValidateCachingIter::checkEquality);
    }

    @NotNull
    static DataSetIterator createTestDataSetIterator(Supplier<MutableInt> state) {
        final int batchSize = 8;
        return new BaseDatasetIterator(batchSize, Integer.MAX_VALUE, new BaseDataFetcher() {
            @Override
            public synchronized void fetch(int numExamples) {
                final double[] features = new double[batchSize];
                final double[][] labels = new double[batchSize][10];
                IntStream.range(0, batchSize).forEach(batch -> {
                    features[batch] = state.get().intValue();
                    labels[batch][state.get().intValue() % 10] = 1;
                    state.get().increment();
                });
                curr = new org.nd4j.linalg.dataset.DataSet(
                        Nd4j.create(features),
                        Nd4j.create(labels));
            }
        });
    }


    private static void performValidation(MiniEpochDataSetIterator iter1, DataSetIterator iter2, int nrofExamplesToTest, BiConsumer<List<DataSet>, List<DataSet>> check) {
        for (int i = 0; i < nrofExamplesToTest; i++) {
            log.info("*************** start round " + i + " ****************");
            List<DataSet> setsCache = new ArrayList<>();
            List<DataSet> setsNoCache = new ArrayList<>();
            while (iter1.hasNext()) {
                setsCache.add(iter1.next());
                setsNoCache.add(iter2.next());
            }
            check.accept(setsCache, setsNoCache);

            iter1.restartMiniEpoch();
            List<DataSet> setsCacheAgain = new ArrayList<>();
            while(iter1.hasNext()) {
                setsCacheAgain.add(iter1.next());
            }
            check.accept(setsCache, setsCacheAgain);

            log.info("Example " + i + " ok!");
            iter1.reset();
        }
    }

    private static void mapFiles(DataProviderBuilder dataProviderBuilder, Path baseDir) {
        try {
            DataSetFileParser.parseFileProperties(baseDir, d -> dataProviderBuilder);
        } catch (IOException e) {
            throw new IllegalArgumentException(e);
        }
    }


    static void checkEquality(List<DataSet> sets1, List<DataSet> sets2) {
        for (DataSet set1 : sets1) {
            final INDArray feats1 = set1.getFeatures();
            final INDArray labs1 = set1.getLabels();
            for (int batch = 0; batch < feats1.size(0); batch++) {
                final INDArray feat1 = getBatchNr(batch, feats1);
                final INDArray lab1 = getBatchNr(batch, labs1);
                boolean match = anyMatch(feat1, lab1, sets2);

                if (!match) {
                    System.out.println("No match found for \n" + feat1);
                    System.out.println("Others: \n" + sets2);
                }
                assertTrue("Features are not the same!", match);
            }
        }
    }

    private static boolean anyMatch(INDArray feat1, INDArray lab1, List<DataSet> sets2) {
        boolean match = false;
        for (DataSet set2 : sets2) {
            final INDArray feats2 = set2.getFeatures();
            final INDArray labs2 = set2.getLabels();
            for (int batch = 0; batch < feats2.size(0); batch++) {
                INDArray batch2 = getBatchNr(batch, feats2);
                INDArray lab2 = getBatchNr(batch, labs2);
                if(lab1.equalsWithEps(lab2, 1e-12) &&
                feat1.equalsWithEps(batch2, 1e-12)) {
                    //System.out.println("found match");
                    log.info("Found match!");
                    match = true;
                    break;
                }
            }
        }
        return match;
    }

    private static INDArray getBatchNr(int batchNr, INDArray batch) {
        // Assume dim 0 is batch dimension
        final int[] dimensions = IntStream.range(1, batch.rank()).toArray();
        return batch.tensorAlongDimension(batchNr, dimensions);
    }

}
