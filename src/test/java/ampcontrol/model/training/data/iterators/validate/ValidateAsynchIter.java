package ampcontrol.model.training.data.iterators.validate;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.Spectrogram;
import ampcontrol.model.training.data.AudioDataProvider;
import ampcontrol.model.training.data.DataProviderBuilder;
import ampcontrol.model.training.data.DataSetFileParser;
import ampcontrol.model.training.data.EvalDataProviderBuilder;
import ampcontrol.model.training.data.iterators.AsynchEnablingDataSetIterator;
import ampcontrol.model.training.data.iterators.Cnn2DDataSetIterator;
import ampcontrol.model.training.data.iterators.DoubleBufferingDataSetIterator;
import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.processing.SilenceProcessor;
import ampcontrol.model.training.data.state.ResetableReferenceState;
import ampcontrol.model.training.data.state.ResetableState;
import ampcontrol.model.training.data.state.ResetableStateFactory;
import org.apache.commons.lang.mutable.MutableInt;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.LocalTime;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

import static ampcontrol.model.training.data.iterators.validate.ValidateCachingIter.createTestDataSetIterator;
import static org.junit.Assert.fail;

/**
 * Test that {@link AsynchEnablingDataSetIterator} with a {@link DoubleBufferingDataSetIterator} does not impact the
 * result of {@link Cnn2DDataSetIterator}.
 *
 * @author Christian Skärby
 */
public class ValidateAsynchIter {

    private static final Logger log = LoggerFactory.getLogger(ValidateAsynchIter.class);

    /**
     * Basically a testcase with an actual data set.
     *
     * @param args well....
     */
    public static void main(String[] args) {

        final int clipLengthMs = 1000;
        final int clipSamplingRate = 44100;
        final Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
        final List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");

        final int timeWindowSize = 100;

        final Supplier<ProcessingResult.Factory> audioPostProcSupplier = () ->
                new Spectrogram(256, 32);

        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, audioPostProcSupplier);
        Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        final ResetableStateFactory stateFactory = new ResetableStateFactory(666);

        final DataProviderBuilder dataProviderBuilder = new EvalDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, stateFactory);
        mapFiles(dataProviderBuilder, baseDir);
        final MiniEpochDataSetIterator iter = getMiniEpochDataSetIterator(new Cnn2DDataSetIterator(dataProviderBuilder.createProvider(), 64, labels), stateFactory);
        performValidation(iter, ValidateAsynchIter::checkSumEquality);
    }

    private static void mapFiles(DataProviderBuilder dataProviderBuilder, Path baseDir) {
        try {
            DataSetFileParser.parseFileProperties(baseDir, d -> dataProviderBuilder);
        } catch (IOException e) {
            throw new IllegalArgumentException(e);
        }
    }

    /**
     * Test that mini epoch can be reset and set. Basically a unit test for some of the functionality of this class
     */
    @Test
    public void resetAndResetMiniEpoch() {
        final ResetableReferenceState<MutableInt> state = new ResetableReferenceState<>(
                mutableInt -> new MutableInt(mutableInt.intValue()),
                new MutableInt(0));

        performValidation(getMiniEpochDataSetIterator(
                createTestDataSetIterator(state),
                state
        ), ValidateCachingIter::checkEquality);
    }

    @NotNull
    private static MiniEpochDataSetIterator getMiniEpochDataSetIterator(
            DataSetIterator sourceIter,
            ResetableState state) {
        final int bufferSize = 5;
        return new AsynchEnablingDataSetIterator(
                new DoubleBufferingDataSetIterator(sourceIter, bufferSize),
                state,
                bufferSize * 3);
    }

    private static void performValidation(MiniEpochDataSetIterator iter, BiConsumer<List<DataSet>, List<DataSet>> compareFun) {

        final List<DataSet> previousMiniEpoch = getDataSets(iter, 0);
        final LocalTime start = LocalTime.now();
        for (int i = 0; i < 5; i++) {
            iter.reset();
            final List<DataSet> miniEpoch = getDataSets(
                    iter, 0);

            iter.restartMiniEpoch();
            final List<DataSet> miniEpochAgain = getDataSets(iter, 0);

            compareFun.accept(miniEpoch, miniEpochAgain);

            final INDArray m1 = getCompareArray(miniEpoch);
            final INDArray prev = getCompareArray(previousMiniEpoch);
            if (prev.equalsWithEps(m1, prev.add(m1).sum().getDouble(0) / 1e5)) {
                log.info("diff: " + prev.sub(m1));
                fail("Produced data was same!!");
            }
            previousMiniEpoch.clear();
            previousMiniEpoch.addAll(miniEpoch);
            log.info("Successful test nr " + i);
        }
        final LocalTime end = LocalTime.now();
        log.info("Time: " + Duration.between(start, end).toMillis());
    }


    @NotNull
    private static List<DataSet> getDataSets(DataSetIterator dataSetIterator, long sleepTime) {
        final List<DataSet> firstMiniEpoch = new ArrayList<>();
        while (dataSetIterator.hasNext()) {

            firstMiniEpoch.add(dataSetIterator.next());
            if (sleepTime > 0) { // Simple model for inference to be able to measure the value of background loading
                try {
                    Thread.sleep(sleepTime);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
        return firstMiniEpoch;
    }

    static void checkSumEquality(List<DataSet> sets1, List<DataSet> sets2)  {
        final INDArray m1 = getCompareArray(sets1);
        final INDArray m2 = getCompareArray(sets2);

        if (!m1.equalsWithEps(m2, m1.add(m2).sum().getDouble(0) / 1e5)) {
            log.info("diff: " + m1.sub(m2));
            System.out.println(m1);
            System.out.println(m2);
            fail("Data was not recreated identically!!");
        }
    }

    private static INDArray getCompareArray(List<DataSet> miniEpoch) {
        // Need to sum up all features in all sets as features typically are entered in random order (even within a batch!)
        return miniEpoch.stream()
                .map(DataSet::getFeatures)
                .map(arr -> arr.mean())
                .reduce(INDArray::add)
                .orElseThrow(() -> new IllegalArgumentException("No data found!!"));
    }
}
