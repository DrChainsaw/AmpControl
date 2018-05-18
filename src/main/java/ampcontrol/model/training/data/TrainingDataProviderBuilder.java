package ampcontrol.model.training.data;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.model.training.data.AudioDataProvider.AudioProcessorBuilder;
import ampcontrol.model.training.data.processing.AudioFileProcessorBuilder;
import ampcontrol.model.training.data.processing.SequentialHoldFileSupplier;
import ampcontrol.model.training.data.processing.WindowedRandomSamplingInfo;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * {@link DataProviderBuilder} for training data. Tries to make many different examples out of the provided files and
 * shuffles the data.
 *
 * @author Christian Skärby
 */
public class TrainingDataProviderBuilder implements DataProviderBuilder {

    private final List<Path> files = new ArrayList<>();
    private final LinkedHashMap<String, AudioProcessorBuilder> labelToBuilder;
    private final Function<Collection<String>, List<String>> labelExpander;
    private final int clipLengthMs;
    private final int windowSizeMs;
    private final Supplier<ProcessingResult.Factory> audioPostProcSupplier;
    private int seed;

    public TrainingDataProviderBuilder(
            Map<String, AudioProcessorBuilder> labelToBuilder,
            Function<Collection<String>, List<String>> labelExpander,
            int clipLengthMs,
            int windowSizeMs,
            Supplier<ProcessingResult.Factory> audioPostProcSupplier,
            int seed) {
        this.labelToBuilder = new LinkedHashMap<>(labelToBuilder);
        this.labelExpander = labelExpander;
        this.clipLengthMs = clipLengthMs;
        this.windowSizeMs = windowSizeMs;
        this.audioPostProcSupplier = audioPostProcSupplier;
        this.seed = seed;
    }

    @Override
    public AudioDataProvider createProvider() {
        return new AudioDataProvider(
                files,
                labelToBuilder,
                new RandomLabelSupplier<>(labelExpander.apply(labelToBuilder.keySet()), new Random(seed++)));
                //new CyclingLabelSupplier<>(labelExpander.apply(labelToBuilder.keySet())));
        //new RandomLabelSupplier<String>(new ArrayList<>(labelToBuilder.keySet()), new Random(seed++)));
    }

    @Override
    public void addLabel(String label) {
        labelToBuilder.put(label, createAudioFileProcessorBuilder());
    }


    @Override
    public DataProviderBuilder addFile(Path file) {
        files.add(file);
        return this;
    }

    @Override
    public int getNrofFiles() {
        return files.size();
    }

    private AudioProcessorBuilder createAudioFileProcessorBuilder() {
        return new AudioFileProcessorBuilder()
                .setSamplingInfoMapper(new WindowedRandomSamplingInfo(clipLengthMs, windowSizeMs, new Random(seed++)))
                .setFileSupplierFactory(new ListShuffler<Path>(new Random(seed++)).andThen(fileList ->  new SequentialHoldFileSupplier(fileList, 1, seed)))
                .setPostProcSupplier(audioPostProcSupplier);
    }
}
