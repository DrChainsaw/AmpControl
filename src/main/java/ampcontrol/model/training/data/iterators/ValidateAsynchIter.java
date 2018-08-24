package ampcontrol.model.training.data.iterators;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.Spectrogram;
import ampcontrol.model.training.data.*;
import ampcontrol.model.training.data.processing.SilenceProcessor;
import ampcontrol.model.training.data.state.ResetableStateFactory;
import org.jetbrains.annotations.NotNull;
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
import java.util.function.Supplier;

/**
 * Test that {@link AsynchEnablingDataSetIterator} with a {@link DoubleBufferingDataSetIterator} does not impact the
 * result of {@link Cnn2DDataSetIterator}. Should ideally be a testcase, but requires a real dataset to run as part of
 * the validation is to see that all parts of the data pipeline can be reset.
 *
 * @author Christian Skärby
 */
public class ValidateAsynchIter {

    private static final Logger log = LoggerFactory.getLogger(ValidateAsynchIter.class);

    public static void main(String[] args) throws InterruptedException {

        int clipLengthMs = 1000;
        int clipSamplingRate = 44100;
        Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
        List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");

        final int timeWindowSize = 100;

        final Supplier<ProcessingResult.Factory> audioPostProcSupplier = () ->
                new Spectrogram(256, 32);

        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, audioPostProcSupplier);
        Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        final ResetableStateFactory stateFactory = new ResetableStateFactory(666);

        final DataProviderBuilder dataProviderBuilder = new EvalDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, stateFactory);
        try {
            DataSetFileParser.parseFileProperties(baseDir, d -> dataProviderBuilder);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        final DataProvider dataProvider = dataProviderBuilder.createProvider();
        final int bufferSize = 5;
        final int batchSize = 64;
        final DataSetIterator sourceIter = new Cnn2DDataSetIterator(dataProvider, batchSize, labels);
        //final MiniEpochDataSetIterator iter = new CachingDataSetIterator(sourceIter, 100);
        final MiniEpochDataSetIterator iter =
                new AsynchEnablingDataSetIterator(
              // sourceIter,
                new DoubleBufferingDataSetIterator(sourceIter, bufferSize).initCache(),
                stateFactory,
                bufferSize*3);
        final long sleepTime = 0;
       final List<DataSet> previousMiniEpoch = getDataSets(iter, sleepTime);
        final LocalTime start = LocalTime.now();
        for (int i = 0; i < 5; i++) {
            iter.reset();

            final List<DataSet> miniEpoch = getDataSets(
                    iter, sleepTime);

            iter.restartMiniEpoch();
            final List<DataSet> miniEpochAgain = getDataSets(iter, sleepTime);

            logDataSet(miniEpoch, "0: ");
            logDataSet(miniEpochAgain, "1: ");

            final INDArray m1 = getCompareArray(miniEpoch);
            final INDArray m2 = getCompareArray(miniEpochAgain);

            if(!m1.equalsWithEps(m2, m1.add(m2).sum().getDouble(0) / 1e5)) {
                log.info("diff: " + m1.sub(m2));
                throw new IllegalStateException("Data was not recreated identically!!");
            }
            final INDArray prev = getCompareArray(previousMiniEpoch);
            if(prev.equalsWithEps(m1,prev.add(m1).sum().getDouble(0) / 1e5)) {
                log.info("diff: " + prev.sub(m1));
                throw new IllegalStateException("Produced data was same!!");
            }
            previousMiniEpoch.clear();
            previousMiniEpoch.addAll(miniEpoch);
            log.info("Successful test nr " + i);
            //log.info("iter " + i + " done");
        }
        final LocalTime end = LocalTime.now();
        log.info("Time: " + Duration.between(start, end).toMillis());
    }

    private static void logDataSet(List<DataSet> miniEpoch, String s) {
        log.info(s + miniEpoch.stream()
                .map(DataSet::getFeatures)
                .map(arr -> arr.mean(0, 1, 2, 3))
                .reduce(INDArray::add));
    }

    @NotNull
    private static List<DataSet> getDataSets(DataSetIterator dataSetIterator, long sleepTime) throws InterruptedException {
        final List<DataSet> firstMiniEpoch = new ArrayList<>();
        while (dataSetIterator.hasNext()) {

            firstMiniEpoch.add(dataSetIterator.next());
            if(sleepTime > 0) { // Simple model for inference to be able to measure the value of background loading
                Thread.sleep(sleepTime);
            }
        }
        return firstMiniEpoch;
    }

    private static INDArray getCompareArray(List<DataSet> miniEpoch) {
        // Need to sum up all features in all sets as features typically are entered in random order (even within a batch!)
        return miniEpoch.stream().map(DataSet::getFeatures).map(arr -> arr.mean()).reduce(INDArray::add).orElseThrow(() -> new IllegalArgumentException("No data found!!"));
    }
}
