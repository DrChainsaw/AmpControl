package ampcontrol.model.training.data.iterators;

import ampcontrol.audio.processing.Pipe;
import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.Spectrogram;
import ampcontrol.audio.processing.UnitMaxZeroMean;
import ampcontrol.model.training.data.AudioDataProvider;
import ampcontrol.model.training.data.DataProviderBuilder;
import ampcontrol.model.training.data.DataSetFileParser;
import ampcontrol.model.training.data.TrainingDataProviderBuilder;
import ampcontrol.model.training.data.processing.SilenceProcessor;
import ampcontrol.model.training.data.state.ResetableStateFactory;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;

public class ValidateAsynchIter {

    private static final Logger log = LoggerFactory.getLogger(ValidateAsynchIter.class);

    public static void main(String[] args) throws InterruptedException {

        int clipLengthMs = 1000;
        int clipSamplingRate = 44100;
        Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
        List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");

        final int timeWindowSize = 100;

        final Supplier<ProcessingResult.Factory> audioPostProcSupplier = () -> new Pipe(
                new Spectrogram(512, 128),
                new UnitMaxZeroMean()
        );

        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, audioPostProcSupplier);
        Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        final ResetableStateFactory stateFactory = new ResetableStateFactory(666);

        final DataProviderBuilder dataProviderBuilder = new TrainingDataProviderBuilder(labelToBuilder, ArrayList::new, clipLengthMs, timeWindowSize, audioPostProcSupplier, stateFactory);
        try {
            DataSetFileParser.parseFileProperties(baseDir, d -> dataProviderBuilder);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        final DataSetIterator sourceIter = new Cnn2DDataSetIterator(dataProviderBuilder.createProvider(), 8, labels);
        final MiniEpochDataSetIterator asynchIter = new AsynchEnablingDataSetIterator(sourceIter, stateFactory, 3);

        final List<DataSet> previousMiniEpoch = getDataSets(asynchIter);
        for (int i = 0; i < 100; i++) {
            asynchIter.reset();
            final List<DataSet> miniEpoch = getDataSets(asynchIter);
            asynchIter.restartMiniEpoch();
            final List<DataSet> miniEpochAgain = getDataSets(asynchIter);
            if(!miniEpoch.equals(miniEpochAgain)) {
                log.info("First: \n" + miniEpoch.get(0).getLabels() + "\nsecond: \n" + miniEpochAgain.get(0).getLabels());
                throw new IllegalStateException("Data was not recreated identically!!");
            }
            if(previousMiniEpoch.equals(miniEpoch)) {
                throw new IllegalStateException("Produced data was same!!");
            }
            previousMiniEpoch.clear();
            previousMiniEpoch.addAll(miniEpoch);
            log.info("Successful test nr " + i);
        }
    }

    @NotNull
    private static List<DataSet> getDataSets(MiniEpochDataSetIterator asynchIter) {
        final List<DataSet> firstMiniEpoch = new ArrayList<>();
        while (asynchIter.hasNext()) {
            firstMiniEpoch.add(asynchIter.next());
        }
        return firstMiniEpoch;
    }
}
