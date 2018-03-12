package ampControl.model.training.data;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

import ampControl.audio.processing.ProcessingResult;
import ampControl.model.training.data.AudioDataProvider.AudioProcessorBuilder;
import ampControl.model.training.data.processing.AudioFileProcessorBuilder;
import ampControl.model.training.data.processing.SequentialHoldFileSupplier;
import ampControl.model.training.data.processing.WindowedConsecutiveSamplingInfo;

/**
 * {@link DataProviderBuilder} for test set used in evaluation. Tries to ensure that all data is "covered" with as
 * few examples as possible.
 *
 * @author Christian Skärby
 */
public class EvalDataProviderBuilder implements DataProviderBuilder {

	private final List<Path> files = new ArrayList<>();
	private final LinkedHashMap<String, AudioProcessorBuilder> labelToBuilder;
	private final Function<Collection<String>, List<String>> labelExpander;
	private final int clipLengthMs;
	private final int windowSizeMs;
	private final Supplier<ProcessingResult.Processing> audioPostProcSupplier;
	private int seed;	

	public EvalDataProviderBuilder(
			Map<String, AudioProcessorBuilder> labelToBuilder,
			Function<Collection<String>, List<String>> labelExpander,
			int clipLengthMs,
			int windowSizeMs,
			Supplier<ProcessingResult.Processing> audioPostProcSupplier,
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
	}

	@Override
	public void addLabel(String label) {
		labelToBuilder.put(label, createAudioFileProcessorBuilder());	
	}


	@Override
	public DataProviderBuilder addFile(Path file) {
		files.add(file); return this;
	}
	
	@Override
	public int getNrofFiles() {
		return files.size();
	}

	private AudioProcessorBuilder createAudioFileProcessorBuilder() {
		return new AudioFileProcessorBuilder()
		.setSamplingInfoMapper(new WindowedConsecutiveSamplingInfo(clipLengthMs, windowSizeMs))
		.setFileSupplierFactory(fileList -> new SequentialHoldFileSupplier(fileList, clipLengthMs / windowSizeMs))
		.setPostProcSupplier(audioPostProcSupplier);
	}
}
